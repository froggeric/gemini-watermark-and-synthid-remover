#!/usr/bin/env python3
"""
Generate reference values for the CoreML SDXL Euler scheduler unit tests.
DEV-ONLY script; never shipped or run at runtime.

This script loads the diffusers EulerDiscreteScheduler with SDXL config,
generates reference arrays (alphas_cumprod, sigmas, timesteps), runs
add_noise and a 3-step trajectory, and dumps them as C++ constexpr arrays.

Run with the conversion venv python:
    /Users/frederic/.cache/wmr/coreml-sdxl-venv/bin/python scripts/gen_coreml_sd_scheduler_refs.py
"""

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

# Fixed seed for reproducibility
SEED = 0
rng = np.random.RandomState(SEED)
torch_rng = torch.Generator()
torch_rng.manual_seed(SEED)

# Load the SDXL pipeline and extract the Euler scheduler
print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
scheduler = pipe.scheduler
print(f"Scheduler type: {type(scheduler).__name__}")
assert type(scheduler).__name__ == "EulerDiscreteScheduler", \
    f"Expected EulerDiscreteScheduler, got {type(scheduler).__name__}"

# Verify the scheduler config matches SDXL defaults
config = scheduler.config
print(f"\nScheduler config:")
print(f"  num_train_timesteps: {config.num_train_timesteps}")
print(f"  beta_start: {config.beta_start}")
print(f"  beta_end: {config.beta_end}")
print(f"  beta_schedule: {config.beta_schedule}")
print(f"  prediction_type: {config.prediction_type}")
print(f"  timestep_spacing: {config.timestep_spacing}")
print(f"  steps_offset: {config.steps_offset}")

# Expected SDXL config
EXPECTED_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1,
}

for key, expected in EXPECTED_CONFIG.items():
    actual = getattr(config, key)
    assert actual == expected, f"Config mismatch: {key} = {actual}, expected {expected}"

print("\nConfig verified against SDXL defaults.")

# Collect reference arrays
refs = {}

# (a) Full alphas_cumprod (1000 floats) and train sigmas (1001)
refs["alphas_cumprod"] = scheduler.alphas_cumprod.cpu().numpy().astype(np.float32)
# Train sigmas: computed from alphas_cumprod, flipped, with final 0
train_sigmas = (((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5).flip(0)
train_sigmas = torch.cat([train_sigmas, torch.zeros(1)])
refs["train_sigmas"] = train_sigmas.cpu().numpy().astype(np.float32)

# (b) Inference sigmas and timesteps for num_inference_steps=4
num_inference_steps = 4
scheduler.set_timesteps(num_inference_steps)
refs["inference_sigmas_4"] = scheduler.sigmas.cpu().numpy().astype(np.float32)
refs["inference_timesteps_4"] = scheduler.timesteps.cpu().numpy().astype(np.float32)

# (c) img2img truncated timesteps for num_inference_steps=8, strength=0.3
num_steps_img2img = 8
strength = 0.3
scheduler.set_timesteps(num_steps_img2img)
t_start = num_steps_img2img - int(num_steps_img2img * strength)
img2img_timesteps = scheduler.timesteps[t_start:]
refs["img2img_timesteps_8_strength_0.3"] = img2img_timesteps.cpu().numpy().astype(np.float32)

# (d) add_noise reference: deterministic sample and noise at timestep=timesteps[0]
scheduler.set_timesteps(num_inference_steps)
timestep_0 = scheduler.timesteps[0]

# Fixed deterministic sample: arange-derived in [-1, 1]
sample_size = 4 * 16 * 16  # 1024
sample_flat = np.linspace(-1.0, 1.0, sample_size, dtype=np.float32)
sample = sample_flat.reshape(1, 4, 16, 16)  # (batch, channels, height, width)
sample_torch = torch.from_numpy(sample).to(dtype=torch.float32)

# Fixed noise with seed
noise_torch = torch.randn(1, 4, 16, 16, generator=torch_rng, dtype=torch.float32)

# Add noise at timestep 0
noisy_sample = scheduler.add_noise(
    original_samples=sample_torch,
    noise=noise_torch,
    timesteps=torch.tensor([timestep_0.item()], dtype=torch.float32)
)
refs["add_noise_output"] = noisy_sample.cpu().numpy().astype(np.float32)
refs["add_noise_sample_input"] = sample_torch.cpu().numpy().astype(np.float32)
refs["add_noise_input"] = noise_torch.cpu().numpy().astype(np.float32)
refs["add_noise_timestep"] = timestep_0.cpu().numpy().astype(np.float32)

# (e) 3-step trajectory: from the noisy sample, run scale_model_input + step with fixed dummy model_output
# Reset scheduler
scheduler.set_timesteps(num_inference_steps)

# Use the noisy sample from (d) as starting point
current_sample = noisy_sample

# Fixed dummy model_output (same seed)
model_output_torch = torch.randn(1, 4, 16, 16, generator=torch_rng, dtype=torch.float32)
refs["step_model_output"] = model_output_torch.cpu().numpy().astype(np.float32)

trajectory = []
for i in range(3):
    step_index = i
    # Scale model input
    scaled_input = scheduler.scale_model_input(current_sample, scheduler.timesteps[step_index])
    trajectory.append(("scaled_input", i, scaled_input.cpu().numpy().astype(np.float32)))

    # Apply Euler step
    step_output = scheduler.step(
        model_output=model_output_torch,
        timestep=scheduler.timesteps[step_index],
        sample=current_sample,
    )
    trajectory.append(("prev_sample", i, step_output.prev_sample.cpu().numpy().astype(np.float32)))
    current_sample = step_output.prev_sample

refs["trajectory"] = trajectory

# Generate C++ header
header_path = "tests/unit/coreml_sd_scheduler_refs.hpp"
print(f"\nWriting reference header to {header_path}...")

with open(header_path, "w") as f:
    f.write("// GENERATED BY scripts/gen_coreml_sd_scheduler_refs.py\n")
    f.write("// DO NOT EDIT - regenerate with the python script\n\n")
    f.write("#pragma once\n\n")
    f.write("#include <cstdint>\n\n")
    f.write("namespace wmr::test::coreml_sd {\n\n")

    # Helper macro for array definition
    def write_array(name, data):
        f.write(f"constexpr const float {name}[] = {{\n")
        # Format with 6 decimal places, 8 values per line
        for i in range(0, len(data), 8):
            row = data[i:i+8]
            f.write("    " + ", ".join(f"{v:.6f}f" for v in row))
            if i + 8 < len(data):
                f.write(",")
            f.write("\n")
        f.write("};\n")
        f.write(f"constexpr size_t {name}_size = {len(data)};\n\n")

    # (a) alphas_cumprod and train sigmas
    write_array("alphas_cumprod", refs["alphas_cumprod"])
    write_array("train_sigmas", refs["train_sigmas"])

    # (b) inference sigmas and timesteps for 4 steps
    write_array("inference_sigmas_4", refs["inference_sigmas_4"])
    write_array("inference_timesteps_4", refs["inference_timesteps_4"])

    # (c) img2img timesteps
    write_array("img2img_timesteps_8_strength_0_3", refs["img2img_timesteps_8_strength_0.3"])

    # (d) add_noise refs
    write_array("add_noise_sample_input", refs["add_noise_sample_input"].flatten())
    write_array("add_noise_input", refs["add_noise_input"].flatten())
    write_array("add_noise_output", refs["add_noise_output"].flatten())
    # timestep is a scalar
    timestep_val = float(refs['add_noise_timestep'])
    f.write(f"constexpr float add_noise_timestep = {timestep_val:.6f}f;\n\n")

    # (e) trajectory refs
    write_array("step_model_output", refs["step_model_output"].flatten())
    for step_name, step_idx, step_data in refs["trajectory"]:
        if step_name == "scaled_input":
            name = f"trajectory_scaled_input_{step_idx}"
        else:  # prev_sample
            name = f"trajectory_prev_sample_{step_idx}"
        write_array(name, step_data.flatten())

    f.write("} // namespace wmr::test::coreml_sd\n")

print(f"Reference header written to {header_path}")
print("\nSummary:")
print(f"  alphas_cumprod: {len(refs['alphas_cumprod'])} values")
print(f"  train_sigmas: {len(refs['train_sigmas'])} values")
print(f"  inference_sigmas_4: {len(refs['inference_sigmas_4'])} values")
print(f"  inference_timesteps_4: {len(refs['inference_timesteps_4'])} values")
print(f"  img2img_timesteps: {len(refs['img2img_timesteps_8_strength_0.3'])} values")
print(f"  add_noise_output: {refs['add_noise_output'].size} values")
print(f"  trajectory: {len(refs['trajectory'])} arrays ({len(refs['trajectory']) // 2} steps)")
