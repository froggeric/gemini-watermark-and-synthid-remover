#!/usr/bin/env python3
"""Build the fixture set for the M0 eval.

AI set: curated Gemini-generated originals from the repo's untracked
reference/test dirs (watermarked originals are fine -- the detectors under
test are whole-image AI detectors, not watermark detectors; but regen
OUTPUTS are excluded, they are a different population). Duplicated files
(same Gemini_Generated_Image id in two dirs) appear once.

Real set: Wikimedia Commons "Quality images" camera photographs downloaded
into fixtures/real/ by the fetch step recorded in fixtures/real/manifest.json
(fetch date 2026-09-01, served as width-2000 JPEG derivatives of the camera
originals; license: Commons quality images are CC or PD -- see manifest
titles for per-file attribution). The downloads are throttled because the
Commons CDN 429s aggressive original-file fetches.

Outputs symlinks fixtures/ai/<slug>.png and validates fixtures/real/*.jpg,
then writes fixtures/manifest.csv (set,slug,source,w,h).
"""

from __future__ import annotations

import csv
import glob
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIX = os.path.join(HERE, "fixtures")

# (slug, repo-relative path) -- all AI-generated originals, no regen outputs,
# no pure-color test plates.
AI_FIXTURES = [
    # 896x1200 Gemini 3.6 portrait set (10 distinct generations)
    *[("gem36-%02d" % (i + 1),
       "reference-images/896x1200-gemini36/%s" % os.path.basename(p))
      for i, p in enumerate(sorted(glob.glob(os.path.join(
          REPO, "reference-images/896x1200-gemini36/Gemini_Generated_Image_*.png"))))],
    ("regen-e3hcto", "reference-images/regen+reconstruct-testing/Gemini_Generated_Image_e3hctoe3hctoe3hc.png"),
    ("regen-iu84ox", "reference-images/regen+reconstruct-testing/Gemini_Generated_Image_iu84oxiu84oxiu84-2.png"),
    ("vae-5nvcl6", "reference-images/vae-testing/Gemini_Generated_Image_5nvcl65nvcl65nvc.png"),
    ("vae-iau18v", "reference-images/vae-testing/Gemini_Generated_Image_iau18viau18viau1.png"),
    ("vae-x1hsoq", "reference-images/vae-testing/Gemini_Generated_Image_x1hsoqx1hsoqx1hs.png"),
    ("wide-test1", "test-images/2400x1792-test1-gemini.png"),
    ("wide-test2", "test-images/2400x1792-test2-gemini.png"),
    ("port-test3", "test-images/896x1200-test3-gemini36.png"),
    ("port-test4", "test-images/896x1200-test4-gemini36.png"),
    ("paid-36", "reference-images/2816x1536-gemini/gemini-36-paid.png"),
    ("paid-pro", "reference-images/2816x1536-gemini/gemini-pro-paid.png"),
    ("paint-fawn", "test-images/paintings/fawn.png"),
    ("paint-lioness", "test-images/paintings/lioness.png"),
    ("paint-cow", "test-images/paintings/highland_cow.png"),
    ("poster-artnight", "test-images/poster-artnight.png"),
]


def main() -> None:
    os.makedirs(os.path.join(FIX, "ai"), exist_ok=True)
    os.makedirs(os.path.join(FIX, "real"), exist_ok=True)

    rows = []
    missing = []

    for slug, rel in AI_FIXTURES:
        src = os.path.join(REPO, rel)
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = os.path.join(FIX, "ai", slug + ".png")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        img = cv2.imread(src)
        rows.append(("ai", slug, rel, img.shape[1], img.shape[0]))

    real = sorted(glob.glob(os.path.join(FIX, "real", "*.jpg")))
    for p in real:
        img = cv2.imread(p)
        if img is None:
            missing.append(p)
            continue
        rows.append(("real", os.path.splitext(os.path.basename(p))[0],
                     "fixtures/real/" + os.path.basename(p),
                     img.shape[1], img.shape[0]))

    with open(os.path.join(FIX, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "slug", "source", "w", "h"])
        w.writerows(rows)

    n_ai = sum(1 for r in rows if r[0] == "ai")
    n_real = sum(1 for r in rows if r[0] == "real")
    print(f"fixtures: {n_ai} AI, {n_real} real")
    if missing:
        print("MISSING:")
        for m in missing:
            print(" ", m)


if __name__ == "__main__":
    main()
