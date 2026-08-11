#!/usr/bin/env python3
"""Phase A regen-model sweep for the SynthID scrub.

Question: does any img2img model clear SynthID at lower perturbation (less lossy)
and/or faster than the SDXL baseline, while fitting modest memory (this machine
is 16 GB, so we test the lightweight/quantized variants that would actually ship)?

For each (model, strength) point on the validated SynthID-bearing image set:
  - downscale the input to ~1024px (model-native), preserving aspect, dims % 64
  - run img2img with an empty prompt (unconditional), fixed seed
  - record diff_mean + PSNR vs the pre-regen input, wall time, and peak RSS
  - save the output image under outputs/<model>/<image>_s<strength>.png

Clearance is NOT measured here (no in-process SynthID verifier exists; Google's
"Verify with SynthID" is manual, ~10/day). This script produces outputs + metrics;
the user batches outputs through the verifier to find each model's clearing point.

Run:
  .venv/bin/python sweep.py                       # full sweep, all models
  .venv/bin/python sweep.py --quick               # 1 image, smoke test
  .venv/bin/python sweep.py --models sdxl sdxl-turbo
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

import numpy as np
import torch
from PIL import Image

from diffusers import AutoPipelineForImage2Image, AutoencoderKL, FluxImg2ImgPipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # gemini-watermark-and-synthid-remover/
OUT = HERE / "outputs"

SEED = 42
NATIVE = 1024          # long-side target; SDXL/Flux/Qwen are 1024-native
ALIGN = 16             # round dims to a multiple of 16. Flux's patchify needs %16
                       # (SDXL only needs %8, and %16 satisfies both), so every model
                       # outputs the SAME dims for a given image -> pixel-aligned
                       # cross-model comparison. Round-to-nearest keeps aspect drift <~1.5%.
SDXL_VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"   # cached; avoids fp16 VAE darkening

# Default input set: SynthID-bearing images (verified to carry the watermark).
DEFAULT_IMAGES = [
    REPO / "reference-images" / "synthid-verified" / "poster-baduanjin.png",
    REPO / "reference-images" / "synthid-verified" / "test-regen003-detected.png",
    REPO / "reference-images" / "synthid-verified" / "test-regen005-cpu.png",
    REPO / "reference-images" / "synthid-verified" / "test-regen005-detected.png",
    REPO / "reference-images" / "synthid-verified" / "test-regen009-detected.png",
    REPO / "images" / "astronaut-synthid.png",
    REPO / "images" / "fisherman-synthid.png",
]


@dataclass
class ModelSpec:
    key: str
    loader: str                         # "pretrained" | "gguf_single_file"
    repo: str                           # HF repo (pretrained) or gguf repo id
    points: list[tuple[float, int]]     # (strength, num_inference_steps)
    guidance: float
    filename: str | None = None         # gguf filename within repo (gguf loader)
    components_repo: str | None = None  # repo to pull encoders/VAE from (gguf loader)
    variant: str | None = None          # e.g. "fp16" for SDXL-family
    needs_fp16_vae_fix: bool = False
    offload: str = "none"               # "none" | "model" | "sequential"
    dtype: torch.dtype | None = None    # None -> float16; Flux uses bfloat16
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "sdxl": ModelSpec(
        key="sdxl",
        loader="pretrained",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        points=[(0.05, 50), (0.08, 50), (0.10, 50), (0.15, 50)],
        guidance=7.5,
        variant="fp16",
        needs_fp16_vae_fix=True,
        offload="none",
        note="baseline; eff steps = int(50*s) = 2,4,5,7",
    ),
    "sdxl-turbo": ModelSpec(
        key="sdxl-turbo",
        loader="pretrained",
        repo="stabilityai/sd-turbo",
        points=[(0.10, 8), (0.25, 8), (0.50, 8)],
        guidance=1.0,                   # distilled, no CFG
        variant="fp16",
        needs_fp16_vae_fix=True,
        offload="none",
        note="1-4 step distilled; eff steps = int(8*s) = 1,2,4",
    ),
    "flux-schnell": ModelSpec(
        key="flux-schnell",
        loader="gguf_single_file",
        # Q5_0 ~7 GB: the quantized variant that fits 16 GB and is what would ship.
        repo="city96/FLUX.1-schnell-gguf",
        filename="flux1-schnell-Q5_0.gguf",
        points=[(0.25, 4), (0.50, 4), (0.75, 4)],
        guidance=0.0,                   # schnell is guidance-free
        offload="model",                # sequential offload breaks on GGUF; model offload fits 16 GB
        dtype=torch.bfloat16,           # Flux is bf16-trained (fp16 -> NaNs)
        note="4-step flow; eff=int(4*s)=1,2,3",
    ),
    "sdxl-lightning": ModelSpec(
        key="sdxl-lightning",
        loader="sdxl_lightning",
        repo="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_8step_unet.safetensors",
        points=[(0.50, 8), (0.875, 8)],   # eff 4, 7 (8-step distilled comfort zone)
        guidance=1.0,                      # distilled, no CFG
        variant="fp16",
        note="8-step distilled SDXL; eff=int(8*s)=4,7. Does it beat Turbo's lossiness?",
    ),
    "qwen-image": ModelSpec(
        key="qwen-image",
        loader="qwen_img2img",
        repo="qwen/Qwen-Image",
        points=[(0.10, 20), (0.25, 20), (0.50, 20)],   # eff 2,5,10
        guidance=4.0,
        dtype=torch.bfloat16,
        offload="sequential",             # non-quantized so sequential works; but full model is large
        note="Qwen-Image full is ~large; may OOM on 16 GB -> needs quantized/bigger box",
    ),
    "sd3.5": ModelSpec(
        key="sd3.5",
        loader="sd3_img2img",
        repo="stabilityai/stable-diffusion-3.5-large",
        points=[(0.10, 28), (0.25, 28), (0.50, 28)],   # eff 3,7,14
        guidance=4.0,
        variant="fp16",
        offload="model",
        note="SD3.5 Large MMDiT; gated? tight on 16 GB",
    ),
    "z-image": ModelSpec(
        key="z-image",
        loader="zimage_img2img",
        repo="Tongyi-MAI/Z-Image-Turbo",
        points=[(0.25, 9), (0.50, 9), (0.875, 9)],   # eff 2,4,7 (turbo: 9 steps = 8 DiT fwd)
        guidance=0.0,                       # turbo: guidance 0
        dtype=torch.bfloat16,
        offload="model",
        note="6B Apache-2.0 turbo; card says txt2img-only but diffusers has Img2Img pipeline",
    ),
    "chroma": ModelSpec(
        key="chroma",
        loader="chroma_gguf",
        repo="silveroxides/Chroma-GGUF",
        filename="Chroma1-HD/Chroma1-HD-Q4_0.gguf",
        components_repo="lodestones/Chroma1-HD",
        points=[(0.10, 40), (0.25, 40), (0.50, 40)],   # eff 4,10,20 (card: 40 steps, guidance 3)
        guidance=3.0,
        dtype=torch.bfloat16,
        offload="model",
        note="8.9B Flux-family, Apache 2.0; Q4 gguf + model offload. Card=t2g but diffusers has Img2Img.",
    ),
    # Follow-on (not in the first run; add when ready):
    # "sdxl-lightning": swap UNet to ByteDance/SDXL-Lightning sdxl_lightning_8step.
    # "qwen-image":     Qwen::QwenImageRunner in sdcpp; diffusers Qwen-Image img2img path unverified.
    # "ctrlregen":      yepengliu/ctrlregen (clean-noise regen reference).
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def peak_mem_mb() -> int:
    # RSS (CPU heap) + MPS-allocated unified memory (where the weights/activations
    # actually live on Apple Silicon). ru_maxrss alone under-reports on MPS.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    mps = 0
    try:
        if torch.backends.mps.is_available():
            mps = int(torch.mps.current_allocated_memory() / (1024 * 1024))
    except Exception:
        pass
    return rss + mps


def _apply_offload(pipe, spec: ModelSpec, device: str):
    if spec.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif spec.offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        try:
            pipe = pipe.to(device)
        except Exception as e:  # OOM / MPS hiccup -> fall back to model offload
            print(f"  [load] .to({device}) failed ({e!r}); using model_cpu_offload")
            pipe.enable_model_cpu_offload()
    return pipe


def load_pipeline(spec: ModelSpec, device: str):
    dtype = spec.dtype or torch.float16
    if spec.loader == "pretrained":
        kw = dict(torch_dtype=dtype, variant=spec.variant) if spec.variant else dict(torch_dtype=dtype)
        if spec.needs_fp16_vae_fix:
            kw["vae"] = AutoencoderKL.from_pretrained(SDXL_VAE_FIX, torch_dtype=dtype)
        pipe = AutoPipelineForImage2Image.from_pretrained(spec.repo, **kw)
    elif spec.loader == "sdxl_lightning":
        # Distilled SDXL (8-step): the ByteDance UNet shard is a raw state_dict (not a
        # from_single_file checkpoint), so load the SDXL pipeline then overwrite the
        # UNet weights. Trailing-step Euler is the schedule Lightning was distilled for.
        from diffusers import EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        vae = AutoencoderKL.from_pretrained(SDXL_VAE_FIX, torch_dtype=dtype)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            variant="fp16", torch_dtype=dtype)
        path = hf_hub_download(repo_id=spec.repo, filename=spec.filename)
        pipe.unet.load_state_dict(load_file(path))   # base UNet <- Lightning distilled
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing")
    elif spec.loader == "qwen_img2img":
        # Qwen-Image base DOES expose a strength-img2img pipeline (not just edit).
        from diffusers import QwenImageImg2ImgPipeline
        pipe = QwenImageImg2ImgPipeline.from_pretrained(spec.repo, torch_dtype=dtype)
    elif spec.loader == "sd3_img2img":
        from diffusers import StableDiffusion3Img2ImgPipeline
        kw = dict(torch_dtype=dtype)
        if spec.variant:
            kw["variant"] = spec.variant
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(spec.repo, **kw)
    elif spec.loader == "chroma_gguf":
        # Chroma (8.9B Flux-family): transformer from gguf + encoders/VAE from the
        # official repo. Same pattern as the Flux gguf loader.
        from diffusers import ChromaTransformer2DModel, ChromaImg2ImgPipeline, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec.repo, filename=spec.filename)
        transformer = ChromaTransformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype)
        pipe = ChromaImg2ImgPipeline.from_pretrained(
            spec.components_repo, transformer=transformer, torch_dtype=dtype)
    elif spec.loader == "zimage_img2img":
        # diffusers ships ZImageImg2ImgPipeline (model-agnostic init path); Tongyi's card
        # calls Turbo txt2img-only, so this is an unverified img2img test.
        from diffusers import ZImageImg2ImgPipeline
        pipe = ZImageImg2ImgPipeline.from_pretrained(spec.repo, torch_dtype=dtype)
    elif spec.loader == "gguf_single_file":
        # GGUF = transformer only. from_single_file mishandles the quantized tensor
        # shapes unless given an explicit GGUFQuantizationConfig. Load the transformer
        # that way, then assemble the pipeline from_pretrained (official repo, now
        # accessible) so encoders + VAE + scheduler come along automatically.
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec.repo, filename=spec.filename)
        transformer = FluxTransformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype)
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=dtype)
    else:
        raise ValueError(f"unknown loader {spec.loader}")
    pipe.set_progress_bar_config(disable=True)
    pipe = _apply_offload(pipe, spec, device)
    return pipe


def _align(v: int) -> int:
    # nearest multiple of ALIGN (>= ALIGN); round-to-nearest minimizes aspect drift.
    return max(ALIGN, (v + ALIGN // 2) // ALIGN * ALIGN)


def prep_image(path: Path) -> Image.Image:
    """Load, downscale to long side NATIVE preserving aspect, align dims to ALIGN.

    ALIGN=16 satisfies both SDXL (%8 VAE) and Flux (%16 patchify), so every model
    outputs identical dims for a given image (pixel-aligned cross-model comparison).
    """
    from PIL import ImageOps
    img = Image.open(path).convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    w, h = img.size
    scale = NATIVE / max(w, h)
    nw, nh = (_align(round(w * scale)), _align(round(h * scale))) if scale < 1.0 else (_align(w), _align(h))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img


def run_point(pipe, spec: ModelSpec, init: Image.Image, strength: float, steps: int,
              generator) -> Image.Image:
    out = pipe(
        prompt="",
        image=init,
        height=init.height,
        width=init.width,        # Flux defaults to 1024x1024; force the input's aspect
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=spec.guidance,
        generator=generator,
    )
    return out.images[0]


def metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """diff_mean (0-255, lower=closer) and PSNR (dB, higher=closer) for a vs b."""
    aa = np.asarray(a.convert("RGB"), dtype=np.float64)
    bb = np.asarray(b.convert("RGB"), dtype=np.float64)
    if aa.shape != bb.shape:
        bb = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS), dtype=np.float64)
    diff = np.abs(aa - bb)
    diff_mean = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return diff_mean, psnr


CSV_FIELDS = ["model", "image", "strength", "steps", "eff_steps",
              "diff_mean", "psnr", "seconds", "peak_mem_mb", "status"]


def _save_csv(rows: list[dict], csv_path: Path) -> None:
    # Rewrite the whole CSV each call (small). Called after every generation so a
    # killed/interrupted run still keeps all completed rows on disk.
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--images", nargs="+", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--quick", action="store_true", help="1 image, 1 strength/model (smoke)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    device = pick_device()
    print(f"device={device}  torch={torch.__version__}  machine_ram_note=16GB")
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    images = [p for p in args.images if p.exists()]
    if not images:
        print("No input images found. Check --images.", file=sys.stderr)
        return 2
    missing = [p for p in args.images if not p.exists()]
    if missing:
        print(f"[warn] missing inputs (skipped): {missing}", file=sys.stderr)

    csv_path = args.out / "metrics.csv"
    _save_csv([], csv_path)   # init header immediately (incremental saves follow)

    for key in args.models:
        if key not in MODELS:
            print(f"[warn] unknown model {key}; skipped", file=sys.stderr)
            continue
        spec = MODELS[key]
        print(f"\n=== loading {key} ({spec.repo}) ===  offload={spec.offload}")
        t0 = time.monotonic()
        try:
            pipe = load_pipeline(spec, device)
        except Exception as e:
            print(f"  [load-failed] {key}: {e!r}", file=sys.stderr)
            continue
        print(f"  loaded in {time.monotonic()-t0:.1f}s  peak_rss={peak_mem_mb()}MB  | {spec.note}")

        model_out = args.out / key
        model_out.mkdir(parents=True, exist_ok=True)

        points = spec.points
        use_images = images[:1] if args.quick else images
        if args.quick:
            points = points[:1]

        # generator lives on CPU when offloading (pipeline moves tensors); else on device
        gen_device = "cpu" if spec.offload != "none" else device

        for img_path in use_images:
            stem = img_path.stem
            try:
                init = prep_image(img_path)
            except Exception as e:
                print(f"  [prep-failed] {stem}: {e!r}", file=sys.stderr)
                continue
            init_path = args.out / f"_input_{stem}_{init.size[0]}x{init.size[1]}.png"
            if not init_path.exists():
                init.save(init_path)

            for (strength, steps) in points:
                eff = max(1, int(round(steps * strength)))
                tag = f"s{strength:.2f}_n{steps}_eff{eff}"
                out_path = model_out / f"{stem}_{tag}.png"
                gen = torch.Generator(device=gen_device).manual_seed(SEED)
                t0 = time.monotonic()
                try:
                    res = run_point(pipe, spec, init, strength, steps, gen)
                except Exception as e:
                    print(f"  [run-failed] {key} {stem} {tag}: {e!r}", file=sys.stderr)
                    rows.append(dict(model=key, image=stem, strength=strength, steps=steps,
                                     eff_steps=eff, diff_mean="", psnr="", seconds="",
                                     peak_mem_mb=peak_mem_mb(),
                                     status=f"error:{type(e).__name__}"))
                    _save_csv(rows, csv_path)
                    continue
                secs = time.monotonic() - t0
                res.save(out_path)
                dm, psnr = metrics(res, init)
                rows.append(dict(model=key, image=stem, strength=strength, steps=steps,
                                 eff_steps=eff, diff_mean=f"{dm:.3f}", psnr=f"{psnr:.2f}",
                                 seconds=f"{secs:.1f}", peak_mem_mb=peak_mem_mb(), status="ok"))
                _save_csv(rows, csv_path)
                print(f"  {stem:30s} {tag:20s} diff={dm:6.2f} psnr={psnr:5.1f}dB "
                      f"{secs:5.1f}s rss={peak_mem_mb()}MB")

        del pipe
        try:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    csv_path = args.out / "metrics.csv"
    _save_csv(rows, csv_path)
    print(f"\nmetrics -> {csv_path}  ({len(rows)} rows)")
    print("outputs ->", args.out)
    print("Next: clear-check the outputs via Google 'Verify with SynthID' to find")
    print("each model's lowest clearing strength; compare diff/PSNR/RSS at that point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
