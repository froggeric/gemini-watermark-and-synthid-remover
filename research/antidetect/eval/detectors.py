#!/usr/bin/env python3
"""Detector wrappers with a uniform interface for the M0 A/B harness.

Each detector implements:

    score(img_rgb_u8) -> float

returning a scalar where HIGHER = "more AI-generated", normalized so that the
detector's own operating threshold is 0.0 (i.e. score > 0 => flagged fake):

  commfor  CommunityForensics ViT-S/16 @ 384 (ONNX, onnxruntime CPU).
           Preprocessing per the FIXED repo preprocessor_config.json
           (2026-07-22 weight regeneration; the older
           CommunityForensics-DeepfakeDet-ViT-ONNX repo is deprecated with
           broken weights -- do not use it):
           shortest-edge 440 resize (bicubic) -> center crop 384 -> CLIP
           normalization. Output: single logit, sigmoid head training
           (num_classes=1); logit > 0 <=> sigmoid > 0.5 <=> AI.
  corvi    Corvi et al. 2022 "GRIP" Grag2021_latent (res50stride1) in
           PyTorch. Full-res input (no resize/crop), ImageNet normalization;
           per-pixel logits averaged over space (grip-unina/DMimageDetection
           test_code/main.py semantics); logit > 0 = fake per the paper.
  npr      NPR (Tan et al. 2024) ResNet-50 variant from
           chuangchuangtan/NPR-DeepfakeDetection (weights NPR.pth in-repo).
           Full-res, ImageNet norm; the network internally computes the
           nearest up/down residual and scales conv1 input by 2/3. Single
           logit; logit > 0 = fake (verified empirically against fixtures).

Score direction for every detector is verified empirically at baseline time
(run_eval asserts AUROC > 0.5 with these conventions on real vs AI fixtures
and reports it).

Run on CPU except corvi/npr torch models, which use MPS when available
(Apple Silicon). corvi inputs are hard-capped at 1.6 MPix (proportional
bicubic) to keep res50stride1 activations inside 16 GB unified memory;
the cap is applied identically to baseline and A/B conditions so deltas
stay comparable (flagged `corvi_capped` in results rows).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
THIRD_PARTY = os.path.join(HERE, "third_party")

# corvi res50stride1 activation budget: at full res layer1 holds 256-channel
# float32 tensors. 1.2 MPix keeps every fixture's forward inside ~4 GB RAM
# (CPU) / MPS wired memory; the 896x1200 Gemini portraits (1.075 MPix) stay
# native. The cap applies identically to baseline and every A/B condition.
CORVI_MAX_PIXELS = 1_200_000

# The torch models run on CPU by default: MPS on this shared dev machine
# stalls for minutes inside MTLCommandBuffer waitUntilCompleted whenever
# another process contends the GPU (measured repeatedly on 2026-09-01).
# Set WMR_EVAL_DEVICE=mps to re-enable.
DEVICE = os.environ.get("WMR_EVAL_DEVICE", "cpu")


def _load_module_from(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. CommunityForensics ViT-S/16 @ 384 (ONNX)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class CommForDetector:
    name = "commfor"
    threshold = 0.0  # logit; sigmoid 0.5
    direction = "higher=fake"

    def __init__(self, onnx_path: str = os.path.join(
            MODELS, "commfor-vit-s16-384.onnx")):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # shortest edge -> 440, PIL bicubic (ANTIALIASED; the canonical
        # transformers/ViTImageProcessor path). Kernel choice matters a lot:
        # cv2.INTER_CUBIC (no downscale antialias) shifts logits by 2-4
        # (see docs/research/antidetect-m0-calibration.md, "resize-kernel
        # sensitivity"); cv2.INTER_AREA is the closest C++-friendly match.
        from PIL import Image
        h, w = img.shape[:2]
        scale = 440.0 / min(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        r = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BICUBIC))
        # center crop 384
        top, left = (nh - 384) // 2, (nw - 384) // 2
        c = r[top:top + 384, left:left + 384]
        x = c.astype(np.float32) / 255.0
        x = (x - CLIP_MEAN) / CLIP_STD
        return np.transpose(x, (2, 0, 1))[None]

    def score(self, img: np.ndarray) -> float:
        x = self.preprocess(img)
        out = self.sess.run(None, {self.in_name: x})[0]
        return float(np.asarray(out).ravel()[0])

    def score_sigmoid(self, img: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-self.score(img))))


# ---------------------------------------------------------------------------
# 2. Corvi GRIP Grag2021_latent (res50stride1, PyTorch)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CorviDetector:
    name = "corvi"
    threshold = 0.0  # logit
    direction = "higher=fake"

    def __init__(self, ckpt: str = os.path.join(
            MODELS, "Grag2021_latent_epoch_best.pth")):
        import torch
        self.torch = torch
        resnet_mod = _load_module_from(
            os.path.join(THIRD_PARTY, "DMimageDetection", "test_code",
                         "networks", "resnet_mod.py"),
            "grip_resnet_mod")
        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        model.load_state_dict(sd)
        model.eval()
        self.device = ("mps" if DEVICE == "mps"
                       and torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        if self.device == "cpu":
            # channels_last is measurably faster for the stride-1 convs
            self.model = self.model.to(memory_format=torch.channels_last)
        self.last_capped = False

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # full-res, ImageNet norm; hard cap for the stride1 activation budget
        x = img
        h, w = x.shape[:2]
        self.last_capped = False
        if h * w > CORVI_MAX_PIXELS:
            s = (CORVI_MAX_PIXELS / (h * w)) ** 0.5
            x = cv2.resize(x, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_CUBIC)
            self.last_capped = True
        x = x.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(x, (2, 0, 1))[None]

    def score(self, img: np.ndarray) -> float:
        torch = self.torch
        x = self.preprocess(img)
        t = torch.from_numpy(x).to(self.device)
        if self.device == "cpu":
            t = t.contiguous(memory_format=torch.channels_last)
        try:
            with torch.no_grad():
                out = self.model(t)
        except RuntimeError as e:  # MPS pressure fallback
            if "out of memory" not in str(e).lower():
                raise
            self.model = self.model.to("cpu")
            self.device = "cpu"
            with torch.no_grad():
                out = self.model(torch.from_numpy(x))
        out = out.detach().cpu().numpy()
        # (1,1,H',W') per-pixel logits -> image logit = spatial mean
        # (grip test_code/main.py: logit1 = np.mean(out_tens, (1, 2)))
        return float(out.reshape(out.shape[0], -1).mean())


# ---------------------------------------------------------------------------
# 3. NPR (Tan et al. 2024)

class NPRDetector:
    name = "npr"
    threshold = 0.0
    direction = "higher=fake"

    def __init__(self, weights: str = os.path.join(
            THIRD_PARTY, "NPR-DeepfakeDetection", "NPR.pth")):
        import torch
        self.torch = torch
        resnet = _load_module_from(
            os.path.join(THIRD_PARTY, "NPR-DeepfakeDetection",
                         "networks", "resnet.py"),
            "npr_resnet")
        model = resnet.resnet50(num_classes=1)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))  # training checkpoint
        if any(k.startswith("module.") for k in sd):  # DataParallel prefix
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.device = ("mps" if DEVICE == "mps"
                       and torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        if self.device == "cpu":
            self.model = self.model.to(memory_format=torch.channels_last)

    def score(self, img: np.ndarray) -> float:
        torch = self.torch
        # the NPR forward's nearest down-up needs even dims (their test code
        # resized datasets; their own crop-to-even is commented out upstream)
        h, w = img.shape[:2]
        img = img[: h & ~1, : w & ~1]
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).to(self.device)
        if self.device == "cpu":
            t = t.contiguous(memory_format=torch.channels_last)
        try:
            with torch.no_grad():
                out = self.model(t)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            self.model = self.model.to("cpu")
            self.device = "cpu"
            with torch.no_grad():
                out = self.model(torch.from_numpy(
                    np.transpose(x, (2, 0, 1))[None]))
        return float(out.detach().cpu().numpy().ravel()[0])


# ---------------------------------------------------------------------------
# registry


def build_detectors(which: str = "commfor,corvi,npr") -> dict:
    dets = {}
    for name in [w.strip() for w in which.split(",") if w.strip()]:
        if name == "commfor":
            dets[name] = CommForDetector()
        elif name == "corvi":
            dets[name] = CorviDetector()
        elif name == "npr":
            dets[name] = NPRDetector()
        else:
            raise ValueError(f"unknown detector {name}")
    return dets


def load_rgb(path: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert img is not None and img.ndim == 3 and img.shape[2] == 3, path
    return img


if __name__ == "__main__":
    import glob
    dets = build_detectors()
    probe_real = sorted(glob.glob(os.path.join(HERE, "fixtures", "real", "*.jpg")))[:3]
    probe_ai = sorted(glob.glob(os.path.join(HERE, "fixtures", "ai", "*.png")))[:3]
    for name, d in dets.items():
        rs = [d.score(load_rgb(p)) for p in probe_real]
        as_ = [d.score(load_rgb(p)) for p in probe_ai]
        print(f"{name}: real mean {np.mean(rs):+.3f}  ai mean {np.mean(as_):+.3f}  "
              f"(direction check: {'OK' if np.mean(as_) > np.mean(rs) else 'REVERSED?'})")
