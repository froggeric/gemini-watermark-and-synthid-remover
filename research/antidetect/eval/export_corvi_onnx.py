#!/usr/bin/env python3
"""Export Corvi GRIP Grag2021_latent (res50stride1) to ONNX with preprocessing
baked into the graph.

Runtime contract of the exported graph (export/corvi-grag2021-latent.onnx):

    input  "image"      float32 (N, 3, H, W), RGB, values in [0, 1],
                        H/W dynamic (full-res; the model is fully
                        convolutional until the spatial mean)
    output "logit"      float32 (N, 1); image-level logit = spatial mean of
                        the per-pixel logits (grip-unina test_code/main.py
                        semantics). logit > 0 => AI-generated (Corvi et al.
                        2022); sigmoid(logit) = p_fake.

Baked into the graph: the [0,1] -> ImageNet normalization (mean/std). NOT
baked: any resize/crop (the reference pipeline uses none -- full-res input)
and the uint8 -> float division (pass float [0,1]).

Verification: torch (CPU) vs onnxruntime logits must match within 1e-3 on
3 fixtures; the script asserts and prints the parity numbers.

Usage: .venv/bin/python export_corvi_onnx.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import detectors
from detectors import IMAGENET_MEAN, IMAGENET_STD, _load_module_from, THIRD_PARTY

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "export", "corvi-grag2021-latent.onnx")


def build_export_module():
    import torch
    import torch.nn as nn

    resnet_mod = _load_module_from(
        os.path.join(THIRD_PARTY, "DMimageDetection", "test_code",
                     "networks", "resnet_mod.py"), "grip_resnet_mod")
    model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    ck = torch.load(os.path.join(HERE, "models",
                                 "Grag2021_latent_epoch_best.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))
    model.eval()

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    class ExportWrapper(nn.Module):
        def __init__(self, net, mean, std):
            super().__init__()
            self.net = net
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            x = (image - self.mean) / self.std
            y = self.net(x)                     # (N, 1, H', W') per-pixel logits
            y = y.reshape(y.shape[0], y.shape[1], -1).mean(-1)  # spatial mean
            return y

    return ExportWrapper(model, mean, std)


def main() -> None:
    import torch
    import onnxruntime as ort

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wrapper = build_export_module().eval()

    # representative input for tracing (896x1200, a native fixture size)
    h, w = 1200, 896
    dummy = torch.rand(1, 3, h, w)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, OUT,
            input_names=["image"], output_names=["logit"],
            dynamic_axes={"image": {0: "n", 2: "h", 3: "w"},
                          "logit": {0: "n"}},
            opset_version=18, do_constant_folding=True)

    # the dynamo exporter writes weights as external data by default; inline
    # them so the artifact is a single self-contained file
    import onnx
    model = onnx.load(OUT)  # pulls external data into memory
    for t in model.graph.initializer:
        t.ClearField("data_location")
    if os.path.exists(OUT + ".data"):
        os.remove(OUT + ".data")
    onnx.save_model(model, OUT, save_as_external_data=False)

    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])

    # parity check on 3 fixtures (mix: AI portrait, AI wide, real camera).
    # Inputs above 1.2 MPix are bicubic-capped exactly like CorviDetector's
    # scoring path (the graph is dynamic; parity at the capped sizes is
    # equally valid and keeps the CPU stride-1 forward tractable).
    checks = [
        ("ai/gem36-01.png", "AI 896x1200"),
        ("ai/wide-test1.png", "AI 2400x1792 (capped)"),
        ("real/commons-00.jpg", "REAL 3474x2316 (capped)"),
    ]
    print(f"exported {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    worst = 0.0
    for rel, label in checks:
        img = detectors.load_rgb(os.path.join(HERE, "fixtures", rel))
        h, w = img.shape[:2]
        if h * w > detectors.CORVI_MAX_PIXELS:
            s = (detectors.CORVI_MAX_PIXELS / (h * w)) ** 0.5
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_CUBIC)
        x = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        with torch.no_grad():
            t_logit = float(wrapper(torch.from_numpy(x)).numpy().ravel()[0])
        o_logit = float(sess.run(None, {"image": x})[0].ravel()[0])
        diff = abs(t_logit - o_logit)
        worst = max(worst, diff)
        print(f"  parity {label:16s} torch {t_logit:+.6f}  onnx {o_logit:+.6f}  "
              f"|d|={diff:.2e}")
    assert worst < 1e-3, f"parity violated: {worst}"
    print(f"ONNX parity OK (worst {worst:.2e} < 1e-3)")


if __name__ == "__main__":
    main()
