#!/usr/bin/env python3
"""Calibration analysis: turn results/*.csv into the dose recommendations.

Prints (and writes results/calibration.md):

  1. dose ladder reference -- what each strength maps to in C++ constants
     (from physics_reference.dose_for_strength, the mirrored C++ ladder)
  2. per-detector baseline operating point (real/AI score distributions,
     margin at threshold 0)
  3. per-op effectiveness ranking at the medium dose (mean AI delta and
     flip counts from ab_summary.csv)
  4. best-condition table: conditions ranked by AI flips below threshold
     with zero new real false positives
  5. a proposed default strength: the stack-* condition with the most AI
     flips subject to no new real FPs and |real_delta| small, with the
     runner-ups printed for the human decision (the final constants are
     transcribed into docs/research/antidetect-m0-calibration.md, which is
     the artifact of record -- this script only surfaces the numbers).

Usage: .venv/bin/python calibrate.py   (after run_eval.py)
"""

from __future__ import annotations

import csv
import os

from physics_reference import dose_for_strength

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main() -> None:
    lines = ["# Calibration inputs (generated)", ""]

    # 1. dose ladder
    lines += ["## Dose ladder (C++ constants per strength)", "",
              "| strength | bilateral_d | noise_a | noise_b | ca_px | vignette_k | jpeg_q | sigma_midtone/255 |",
              "|---|---|---|---|---|---|---|---|"]
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = dose_for_strength(s)
        sig = (0.25 * d.noise_a + d.noise_b) ** 0.5 * 255
        lines.append(f"| {s:.2f} | {d.bilateral_d:.1f} | {d.noise_a:.2e} | "
                     f"{d.noise_b:.2e} | {d.ca_px:.2f} | {d.vignette_k:.2f} | "
                     f"U[{d.jpeg_q_lo},{d.jpeg_q_hi}] | {sig:.2f} |")
    lines.append("")

    base = list(csv.DictReader(open(os.path.join(RES, "baseline.csv"))))
    ab = list(csv.DictReader(open(os.path.join(RES, "ab_summary.csv"))))
    per = list(csv.DictReader(open(os.path.join(RES, "per_image.csv"))))

    # 2. baseline operating points
    lines += ["## Baseline operating points (threshold = 0)", "",
              "| detector | AI mean | AI min | real mean | real max | AUROC | ai@0 | real@0 |",
              "|---|---|---|---|---|---|---|---|"]
    for b in base:
        lines.append(f"| {b['detector']} | {b['ai_mean']} | {b['ai_min']} | "
                     f"{b['real_mean']} | {b['real_max']} | {b['auroc']} | "
                     f"{b['ai_flagged@0']}/{b['n_ai']} | {b['real_flagged@0']}/{b['n_real']} |")
    lines.append("")

    # real-native vs real-rescaled NPR split (resampling-sensitive detectors)
    nat = [r for r in per if r["condition"] == "baseline" and r["native_real"] == "yes"]
    resc = [r for r in per if r["condition"] == "baseline" and r["native_real"] == "no"]
    if nat and resc:
        lines += ["## Real subset: native camera originals vs rescaled derivatives", "",
                  "| detector | native mean (n) | rescaled mean (n) |", "|---|---|---|"]
        for d in sorted({r["detector"] for r in nat}):
            nm = [float(r["score"]) for r in nat if r["detector"] == d]
            rm = [float(r["score"]) for r in resc if r["detector"] == d]
            lines.append(f"| {d} | {sum(nm)/len(nm):+.2f} ({len(nm)}) | "
                         f"{sum(rm)/len(rm):+.2f} ({len(rm)}) |")
        lines.append("")

    # 3. per-op effectiveness at medium dose
    lines += ["## Per-op effectiveness (medium dose: strength 0.5 / q92 / each kernel)", "",
              "| condition | detector | AI delta | AI flipped below | real delta | real new FP |",
              "|---|---|---|---|---|---|"]
    med = ["bilateral-0.50", "noise-0.50", "ca-0.50", "vignette-0.50", "jpeg-92",
           "mosaic-malvar5x5", "mosaic-bilinear", "mosaic-edgeaware"]
    for c in med:
        for r in ab:
            if r["condition"] == c:
                lines.append(f"| {c} | {r['detector']} | {r['ai_delta']} | "
                             f"{r['ai_flipped_below']} | {r['real_delta']} | {r['real_new_fp']} |")
    lines.append("")

    # 4. best conditions: AI flips with no new real FPs
    lines += ["## Conditions ranked by AI flips (zero new real FPs only)", ""]
    for d in sorted({r["detector"] for r in ab}):
        cand = [r for r in ab if r["detector"] == d and int(r["real_new_fp"]) == 0]
        cand.sort(key=lambda r: -int(r["ai_flipped_below"]))
        top = [r for r in cand if int(r["ai_flipped_below"]) > 0][:5]
        if not top:
            lines.append(f"- **{d}**: no condition flips AI below threshold at zero real-FP cost")
            continue
        lines.append(f"- **{d}**: " + "; ".join(
            f"{r['condition']} (flips {r['ai_flipped_below']}, ai_delta {r['ai_delta']})"
            for r in top))
    lines.append("")

    # 5. stack ladder table
    lines += ["## Full-stack ladder", "",
              "| condition | detector | AI below th | AI flipped | real new FP |",
              "|---|---|---|---|---|"]
    for c in ("stack-0.25", "stack-0.50", "stack-0.75", "stack-1.00",
              "survival-q90", "survival-q75", "survival-q50"):
        for r in ab:
            if r["condition"] == c:
                lines.append(f"| {c} | {r['detector']} | {r['ai_below_th']} | "
                             f"{r['ai_flipped_below']} | {r['real_new_fp']} |")
    lines.append("")

    out = "\n".join(lines) + "\n"
    open(os.path.join(RES, "calibration.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
