# SynthID spectral codebook is not viable (dot artifact investigation)

Date: 2026-08-01. Branch: `synthid-correctness-pack`. Follows `synthid-codebook-vs-codebook-free-gate.md`.

## Symptom
`wmr synthid <img> --codebook <cb>` left a few highly visible luminous dots, always in the same position, more visible at aggressive strength than gentle.

## Root cause: visible-diamond leakage into the codebook
The codebook was built from raw `gemini_black` HF images, which still contain the **visible** Gemini diamond. Averaging many such captures accumulates the diamond's spectrum coherently (it is at a fixed position), so the codebook's dominant signal is the diamond, not the invisible carrier.

Measured on a held-out `gemini_black` image (1024x1024), codebook path at strength 1.0: the residual (`output - input`) had mean |r| **30.6 in the diamond region vs 0.4 everywhere else**. The subtractor was essentially imprinting the diamond back as the "dots". Fixed position (the diamond is fixed) and strength-scaling (the removal factor) follow directly.

## Fix verified
Building the codebook from diamond-**blanked** training images (the diamond region zeroed) drops the diamond-region residual **30.6 -> 4.1** and the global residual std 3.56 -> 0.84. The dots vanish. So "build from diamond-removed images" is the real fix for the artifact.

## Deeper finding: a clean codebook is useless
A diamond-free codebook is nearly inert (residual std 0.84, does almost nothing). On solid-color captures the only signal above the noise floor is the visible diamond; the invisible SynthID-Image carrier sits at ~0.025/255, below the noise floor. Therefore a fixed spectral codebook is either:
- **diamond-contaminated** (visible dots), or
- **cleaned** (no dots, but near-useless for suppression).

There is no good codebook to ship. The `--codebook-free` path (NoiseResidualSubtractor) does not imprint dots, so the default path is clean.

## Conclusion
Drop the ship-a-codebook workstream. A fixed spectral codebook cannot capture the invisible SynthID-Image carrier (too weak, content-conditional); this is the spectral approach's ceiling, consistent with the broader research that only diffusion regeneration can suppress SynthID on content. (This likely affects reverse-SynthID's shipped codebook for the same reason.)

## Separate gap surfaced
`wmr visible` could not remove the visible diamond from these 1024x1024 `gemini_black` images (geometry detection did not cover that resolution/position; cleaned sample retained ~828 vs 852 bright pixels). That is a visible-mark-remover coverage gap, independent of the dot artifact.
