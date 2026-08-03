#!/usr/bin/env python3
"""
Generate and bake SDXL empty-prompt text embeddings for CoreML UNet.

Dev-only script; never run at runtime. The baked .bin is loaded by the
C++ CoreMLSDPipeline at startup. Verified against the converted UNet's
input layout (2,2048,1,77) fp16.
"""

import os
import sys
import numpy as np
import coremltools as ct
from pathlib import Path
import torch


def main():
    models_dir = Path(os.environ.get("WMR_COREML_SD_MODELS_DIR",
                                    Path.home() / ".cache" / "wmr" / "coreml-sdxl"))
    output_path = models_dir / "empty_prompt_embeds.bin"

    if not models_dir.exists():
        print(f"Error: models directory not found: {models_dir}")
        sys.exit(1)

    # Load diffusers SDXL pipeline (cached; torch_dtype=torch.float16 for fp16 encode)
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        print("Error: diffusers not installed. Install with: pip install diffusers torch")
        sys.exit(1)

    print("Loading SDXL pipeline (cached, fp16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Encode empty prompt exactly as diffusers does for CFG
    # pipe.encode_prompt returns (prompt_embeds, negative_prompt_embeds, prompt_pooled, negative_pooled)
    # For CFG with empty prompt, we use the unconditional (negative) embeds
    print("Encoding empty prompt...")
    prompt_embeds, negative_prompt_embeds, _, _ = pipe.encode_prompt(
        prompt="",
        prompt_2="",
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        negative_prompt_2="",
    )

    # negative_prompt_embeds is the unconditional embeds
    # Shape: (1, 77, 2048) fp16 (concatenation of text_encoder and text_encoder_2)
    # For CFG, we duplicate to batch 2 (uncond, uncond)
    uncond_single = negative_prompt_embeds.detach().cpu().numpy()  # (1, 77, 2048)

    # Check stats before duplication
    print(f"Single embeds stats: mean={float(uncond_single.astype(np.float32).mean()):.6f}, "
          f"std={float(uncond_single.astype(np.float32).std()):.6f}, "
          f"min={float(uncond_single.astype(np.float32).min()):.6f}, "
          f"max={float(uncond_single.astype(np.float32).max()):.6f}")

    # Check for NaN or Inf
    has_nan = np.any(np.isnan(uncond_single.astype(np.float32)))
    has_inf = np.any(np.isinf(uncond_single.astype(np.float32)))
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    uncond_embeds = np.concatenate([uncond_single, uncond_single], axis=0).astype(np.float16)  # (2, 77, 2048)

    print(f"Unconditional embeds shape (single): {uncond_single.shape}")
    print(f"Unconditional embeds shape (duplicated for CFG): {uncond_embeds.shape}")
    print(f"Unconditional embeds dtype: {uncond_embeds.dtype}")

    # Verify shape matches SDXL CFG expectations
    assert uncond_embeds.shape == (2, 77, 2048), f"Unexpected shape: {uncond_embeds.shape}"
    assert uncond_embeds.dtype == np.float16, f"Unexpected dtype: {uncond_embeds.dtype}"

    # Transpose to UNet input layout: (2, 2048, 1, 77) = (batch, hidden, 1, seq)
    # Diffusers feeds (B, seq_len, hidden_dim) but the converted UNet expects
    # (B, hidden_dim, 1, seq_len). The '1' is a dummy dimension.
    uncond_embeds_transposed = uncond_embeds.transpose(0, 2, 1).reshape(2, 2048, 1, 77)

    print(f"Transposed embeds shape: {uncond_embeds_transposed.shape}")

    # Print summary stats for auditing (use float32 for stats to avoid fp16 overflow)
    mean_f32 = float(uncond_embeds_transposed.astype(np.float32).mean())
    std_f32 = float(uncond_embeds_transposed.astype(np.float32).std())
    min_f32 = float(uncond_embeds_transposed.astype(np.float32).min())
    max_f32 = float(uncond_embeds_transposed.astype(np.float32).max())
    print(f"Embeds stats: mean={mean_f32:.6f}, std={std_f32:.6f}, min={min_f32:.6f}, max={max_f32:.6f}")

    # Print first few values for sanity check
    print(f"First 10 values: {uncond_embeds_transposed.ravel()[:10]}")

    # Write raw fp16 bytes
    expected_bytes = 2 * 2048 * 1 * 77 * 2  # batch=2, hidden=2048, dummy=1, seq=77, 2 bytes per fp16
    assert uncond_embeds_transposed.nbytes == expected_bytes, \
        f"Size mismatch: {uncond_embeds_transposed.nbytes} vs {expected_bytes}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uncond_embeds_transposed.tobytes())

    print(f"Wrote embeddings to: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")

    # Note: Validation of the embeddings layout happens in the C++ smoke test
    # (coreml_sd_pipeline_smoke_test.cpp). The Python script only bakes the embeds.
    # The ObjC++ pipeline will load the baked embeds and run the full img2img.
    print("\nEmbeddings baked successfully.")
    print("Validation will be performed by the C++ smoke test (coreml_sd_pipeline_smoke_test.cpp).")


if __name__ == "__main__":
    main()
