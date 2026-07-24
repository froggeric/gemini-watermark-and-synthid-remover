# Gemini / Veo watermark alpha masks

These are the **alpha masks** `wmr` uses to remove the visible Gemini/Veo watermarks by exact
reverse alpha-blend. Provided as standalone files for the community (to inspect, reuse, or build
on). The same masks are compiled into the binary as byte arrays in
[`../embedded_assets.hpp`](../embedded_assets.hpp); the PNGs here are the human-readable form.

## Convention

Each PNG is a **single-channel (grayscale) image** whose pixel value is `alpha * 255`:

- `255` (white) = the watermark is fully opaque there
- `0` (black) = no watermark

Removal inverts the alpha-blend exactly, per channel:

```
original = (watermarked - alpha * 255) / (1 - alpha)
```

where `alpha` is the pixel value divided by 255. The mark is placed bottom-right; for the
small (48px) Gemini 3.6 mark the margin is **(96, 96)** from the bottom-right corner (so the
top-left is at `(width - 48 - 96, height - 48 - 96)`), e.g. `(752, 1056)` on an 896x1200 still,
`(576, 1136)` on a 720x1280 video, `(1136, 576)` on a 1280x720 video. Size scales with output
resolution: 48px small (short side ≤ 1024), 96px large.

## Files

| File | What it is | Used for |
|------|------------|----------|
| `gemini-3.6-diamond-48px.png` | **Gemini 3.6 (Flash) 48px diamond** — the average of 10 distinct captures, each with the mark on a near-uniform-black patch (peak ~0.30). | **Removal** of the Gemini 3.6 mark (both stills and video) + the detection template. |
| `gemini-3.6-diamond-48px-extracted-video-REFERENCE.png` | Pristine video alpha extracted from 3 clean 1280x720 Gemini 3.6 videos (360 frames, mark on a uniform dark patch). | Reference / validation. **Identical to the 48px mask above** (per-pixel std 0.002, shape diff 0.3%, correlation 0.9999) — confirming still and video render the mark at the same ~0.30 alpha. |
| `gemini-3.5-diamond-36px.png` | Gemini 3.5 still small diamond (36px). | Removal of the Gemini 3.5 mark. |
| `gemini-diamond-96px.png` | Large-output diamond (96px). | Removal + detection template for large images/videos. |
| `veo-text-68px.png`, `veo-text-99px.png` | Veo legacy "Veo" text watermark (68x30 and 99x43). | Removal of the Veo legacy text mark. |

## Notes

- These PNGs are **derived** masks: `alpha = max(R,G,B)/255` of the source capture, then
  background-corrected (subtract the 25th-percentile border estimate; a no-op for captures on
  pure black). The source captures live in `embedded_assets.hpp`.
- The Gemini 3.6 mark is rendered at the **same ~0.30 alpha** for stills and video (a pristine
  360-frame measurement confirms this; an earlier "video is stronger" reading was a measurement
  artifact from busy footage).
- SynthID (the *invisible* watermark) is a separate, frequency-domain signal and is not
  represented here.
