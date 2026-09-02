#!/usr/bin/env python3
"""M0 eval driver: detector baselines, per-ingredient A/B, JPEG survival.

Runs every fixture (AI + real) through every condition, scores with every
detector, and writes:

  results/per_image.csv     one row per (fixture, condition, detector)
  results/op_stats.csv      the actual random draws per (fixture, condition)
  results/baseline.csv      real-vs-AI baseline table (mean/AUROC/acc@0)
  results/ab_summary.csv    per-condition detector movement + flip counts
  results/summary.md        the same tables as markdown

Conditions (26):
  baseline
  bilateral-{0.25,0.5,1.0}   single op at ladder doses
  noise-{0.25,0.5,1.0}
  ca-{0.25,0.5,1.0}
  vignette-{0.25,0.5,1.0}
  jpeg-{96,92,88}            single JPEG cycle, fixed q
  mosaic-{malvar5x5,bilinear,edgeaware}   mosaic+demosaic, fixed kernel
  stack-{0.25,0.5,0.75,1.0}  full Stage A, random kernel per seed
  survival-{q90,q75,q50}     stack-0.5 output re-encoded at 90/75/50

Every condition applies to BOTH the AI and the real set; deltas are per-image
vs that fixture's own baseline score. Seeds are stable: seed = crc32(slug +
condition), so the run is fully reproducible.

Usage: .venv/bin/python run_eval.py [--detectors commfor,corvi,npr]
                                     [--sets ai,real] [--out results]
Runtime: ~20 min on an M4 (commfor CPU, corvi/npr MPS).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import zlib

import cv2
import numpy as np

import detectors
from physics_reference import StageAConfig, apply_physics, camera_jpeg_cycle, measure_perturbation

HERE = os.path.dirname(os.path.abspath(__file__))

PER_OP = dict(  # condition suffix -> StageAConfig kwargs
    bilateral=lambda s: dict(bilateral=True, noise=False, mosaic=False, ca=False,
                             vignette=False, jpeg_cycle=False),
    noise=lambda s: dict(bilateral=False, noise=True, mosaic=False, ca=False,
                         vignette=False, jpeg_cycle=False),
    ca=lambda s: dict(bilateral=False, noise=False, mosaic=False, ca=True,
                      vignette=False, jpeg_cycle=False),
    vignette=lambda s: dict(bilateral=False, noise=False, mosaic=False, ca=False,
                            vignette=True, jpeg_cycle=False),
)
STRENGTHS = ("0.25", "0.50", "1.00")
JPEG_DOSES = {"96": 96, "92": 92, "88": 88}
KERNELS = ("malvar5x5", "bilinear", "edgeaware")
STACKS = ("0.25", "0.50", "0.75", "1.00")
SURVIVAL = {"q90": 90, "q75": 75, "q50": 50}


def conditions() -> list[tuple[str, callable]]:
    conds = [("baseline", None)]
    for op in PER_OP:
        for s in STRENGTHS:
            conds.append((f"{op}-{s}", ("op", op, float(s))))
    for q in JPEG_DOSES:
        conds.append((f"jpeg-{q}", ("jpeg", JPEG_DOSES[q])))
    for k in KERNELS:
        conds.append((f"mosaic-{k}", ("kernel", k)))
    for s in STACKS:
        conds.append((f"stack-{s}", ("stack", float(s))))
    for q in SURVIVAL:
        conds.append((f"survival-{q}", ("survival", float(STACKS[1]), SURVIVAL[q])))
    return conds


def make_condition_image(img: np.ndarray, spec, seed: int, ladder: str = "calibrated"):
    """Returns (image, stats) for a condition spec."""
    if spec is None:
        return img, {}
    kind = spec[0]
    if kind == "op":
        _, op, s = spec
        cfg = StageAConfig(strength=s, ladder=ladder, **PER_OP[op](s))
        return apply_physics(img, cfg, seed)
    if kind == "jpeg":
        out = camera_jpeg_cycle(img, spec[1])
        return out, {"jpeg_q": spec[1]}
    if kind == "kernel":
        cfg = StageAConfig(strength=1.0, bilateral=False, noise=False, ca=False,
                           vignette=False, jpeg_cycle=False, kernel=spec[1])
        return apply_physics(img, cfg, seed)
    if kind == "stack":
        return apply_physics(img, StageAConfig(strength=spec[1], ladder=ladder), seed)
    if kind == "survival":
        mid, _ = apply_physics(img, StageAConfig(strength=spec[1], ladder=ladder), seed)
        out = camera_jpeg_cycle(mid, spec[2])
        return out, {"jpeg_q": spec[2], "after_stack": spec[1]}
    raise ValueError(spec)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), labels 1 = positive (AI)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(scores) + 1, dtype=np.float64)
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    sum_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_fixtures(sets: list[str]) -> list[dict]:
    rows = []
    with open(os.path.join(HERE, "fixtures", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            if r["set"] not in sets:
                continue
            rows.append(r)
    # flag native (full-size camera original) vs rescaled Commons derivatives
    commons = {}
    man = os.path.join(HERE, "fixtures", "real", "manifest.json")
    if os.path.exists(man):
        import json
        for m in json.load(open(man)):
            commons[os.path.basename(m["file"]).split(".")[0]] = m
    for r in rows:
        r["native_real"] = ""
        if r["set"] == "real" and r["slug"] in commons:
            r["native_real"] = "yes" if "served_width" not in commons[r["slug"]] else "no"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="commfor,corvi,npr")
    ap.add_argument("--sets", default="ai,real")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--resume", action="store_true",
                    help="skip fixtures already present in per_image.csv")
    ap.add_argument("--legacy-stack", action="store_true",
                    help="use the PRE-calibration dose ladder (noise at every "
                         "strength, CA, vignette, 3-kernel draw) — reproduces "
                         "the historical stack-*/noise-*/ca-*/vignette-* rows "
                         "in results/per_image.csv; default is the calibrated "
                         "ladder that matches C++ dose_for_strength (c5a551a)")
    args = ap.parse_args()
    ladder = "legacy" if args.legacy_stack else "calibrated"
    os.makedirs(args.out, exist_ok=True)

    dets = detectors.build_detectors(args.detectors)
    fixtures = load_fixtures(args.sets.split(","))
    conds = conditions()
    det_names = list(dets)

    per_path = os.path.join(args.out, "per_image.csv")
    stat_path = os.path.join(args.out, "op_stats.csv")
    done_slugs = set()
    if args.resume and os.path.exists(per_path):
        done_slugs = {r["slug"] for r in csv.DictReader(open(per_path))}
        print(f"resume: {len(done_slugs)} fixtures already scored")

    print(f"{len(fixtures)} fixtures x {len(conds)} conditions x "
          f"{len(det_names)} detectors = "
          f"{len(fixtures) * len(conds) * len(det_names)} scores")

    per_fields = ["set", "slug", "native_real", "condition", "detector",
                  "score", "perturbation_255", "corvi_capped"]
    stat_fields = ["slug", "condition", "seed", "kernel", "ca_dx", "ca_dy",
                   "jpeg_q", "perturbation_255"]
    mode = "a" if args.resume else "w"
    f_per = open(per_path, mode, newline="")
    f_stat = open(stat_path, mode, newline="")
    w_per = csv.DictWriter(f_per, fieldnames=per_fields,
                           extrasaction="ignore")
    w_stat = csv.DictWriter(f_stat, fieldnames=stat_fields)
    if mode == "w":
        w_per.writeheader()
        w_stat.writeheader()

    t0 = time.time()
    ndone = 0
    for fi, fx in enumerate(fixtures):
        if fx["slug"] in done_slugs:
            continue
        img = detectors.load_rgb(os.path.join(
            HERE, "fixtures", fx["set"], fx["slug"] + ".png"
            if fx["set"] == "ai" else fx["slug"] + ".jpg"))
        for cname, spec in conds:
            seed = zlib.crc32((fx["slug"] + "|" + cname).encode()) & 0xFFFFFFFF
            cimg, stats = make_condition_image(img, spec, seed, ladder=ladder)
            pert = measure_perturbation(img, cimg)
            for dn in det_names:
                d = dets[dn]
                s = d.score(cimg)
                w_per.writerow({
                    "set": fx["set"], "slug": fx["slug"],
                    "native_real": fx.get("native_real", ""),
                    "condition": cname, "detector": dn, "score": f"{s:.6f}",
                    "perturbation_255": f"{pert:.4f}",
                    "corvi_capped": getattr(d, "last_capped", False) and dn == "corvi",
                })
                if dn == det_names[0]:
                    w_stat.writerow({
                        "slug": fx["slug"], "condition": cname, "seed": seed,
                        "kernel": stats.get("kernel", ""),
                        "ca_dx": f"{stats.get('ca_dx', float('nan')):.4f}",
                        "ca_dy": f"{stats.get('ca_dy', float('nan')):.4f}",
                        "jpeg_q": stats.get("jpeg_q", ""),
                        "perturbation_255": f"{pert:.4f}",
                    })
        f_per.flush()
        f_stat.flush()
        ndone += 1
        el = time.time() - t0
        print(f"[{fi + 1}/{len(fixtures)}] {fx['set']}/{fx['slug']} "
              f"({el:.0f}s, eta {el / max(ndone, 1) * (len(fixtures) - fi - 1):.0f}s)",
              flush=True)

    f_per.close()
    f_stat.close()
    summarize(args.out)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}")


def summarize(outdir: str) -> None:
    """Build baseline.csv, ab_summary.csv, summary.md from per_image.csv."""
    import collections
    rows = list(csv.DictReader(open(os.path.join(outdir, "per_image.csv"))))
    for r in rows:
        r["score"] = float(r["score"])

    def get(cond, d, s):
        return [r for r in rows if r["condition"] == cond
                and r["detector"] == d and r["set"] == s]

    dets = sorted({r["detector"] for r in rows})
    conds = [c for c in dict.fromkeys(r["condition"] for r in rows)]

    # ---- baseline table
    base_rows = []
    for d in dets:
        ai = np.array([r["score"] for r in get("baseline", d, "ai")])
        re = np.array([r["score"] for r in get("baseline", d, "real")])
        labels = np.concatenate([np.ones(len(ai)), np.zeros(len(re))])
        scores = np.concatenate([ai, re])
        base_rows.append({
            "detector": d, "n_ai": len(ai), "n_real": len(re),
            "ai_mean": f"{ai.mean():.3f}", "ai_min": f"{ai.min():.3f}",
            "real_mean": f"{re.mean():.3f}", "real_max": f"{re.max():.3f}",
            "auroc": f"{auroc(labels, scores):.4f}",
            "ai_flagged@0": int((ai > 0).sum()),
            "real_flagged@0": int((re > 0).sum()),
            "acc@0": f"{(np.concatenate([ai > 0, re <= 0]).mean()):.3f}",
        })
    with open(os.path.join(outdir, "baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)

    # ---- A/B movement table
    ab_rows = []
    for c in conds:
        if c == "baseline":
            continue
        for d in dets:
            ai0 = {r["slug"]: r["score"] for r in get("baseline", d, "ai")}
            re0 = {r["slug"]: r["score"] for r in get("baseline", d, "real")}
            ai1 = get(c, d, "ai")
            re1 = get(c, d, "real")
            if not ai1:
                continue
            ai_scores = np.array([r["score"] for r in ai1])
            re_scores = np.array([r["score"] for r in re1])
            deltas_ai = np.array([r["score"] - ai0[r["slug"]] for r in ai1])
            deltas_re = np.array([r["score"] - re0[r["slug"]] for r in re1])
            flips = sum(1 for r in ai1
                        if ai0[r["slug"]] > 0 and r["score"] <= 0)
            new_fp = sum(1 for r in re1
                         if re0[r["slug"]] <= 0 and r["score"] > 0)
            ab_rows.append({
                "condition": c, "detector": d,
                "ai_mean": f"{ai_scores.mean():.3f}",
                "ai_delta": f"{deltas_ai.mean():+.3f}",
                "ai_below_th": int((ai_scores <= 0).sum()),
                "ai_flipped_below": flips,
                "real_mean": f"{re_scores.mean():.3f}",
                "real_delta": f"{deltas_re.mean():+.3f}",
                "real_above_th": int((re_scores > 0).sum()),
                "real_new_fp": new_fp,
            })
    with open(os.path.join(outdir, "ab_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ab_rows[0]))
        w.writeheader()
        w.writerows(ab_rows)

    # ---- markdown
    def md_table(hdr, data_rows):
        out = ["| " + " | ".join(hdr) + " |",
               "|" + "|".join(["---"] * len(hdr)) + "|"]
        for dr in data_rows:
            out.append("| " + " | ".join(str(dr[h]) for h in hdr) + " |")
        return "\n".join(out)

    lines = ["# M0 eval summary (generated; see docs/research/antidetect-m0-calibration.md)",
             "", "## Baseline (real vs AI, no ops)", ""]
    lines.append(md_table(list(base_rows[0]), base_rows))
    lines += ["", "## Per-condition movement (deltas vs each fixture's baseline)", ""]
    lines.append(md_table(list(ab_rows[0]), ab_rows))
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
