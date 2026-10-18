#!/usr/bin/env python3
"""Deterministic summary of measured multiobjective MovieLens JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from html import escape

METRICS = ("mae", "rmse", "precision_at_k", "recall_at_k")
VARIANTS = ("fixed_kurcf", "mae_selected", "recall_selected",
            "unclustered_hybrid", "usercf")
LABELS = ("Fixed KUR-CF", "Validation MAE selection", "Validation Recall selection",
          "Unclustered hybrid", "Pearson UserCF")
COLORS = ("#527aaf", "#13847c", "#b77a36", "#8961aa", "#78879b")


def stats(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite metric")
    rng = random.Random(20261018)
    means = sorted(statistics.mean([values[rng.randrange(len(values))]
                                   for _ in values]) for _ in range(4000))
    def pct(q):
        pos = (len(means) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return means[lo] + (pos - lo) * (means[hi] - means[lo])
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ci95": [pct(.025), pct(.975)], "n_seeds": len(values)}


def aggregate(runs: list[dict], min_seeds: int = 5) -> dict:
    if len(runs) < min_seeds:
        raise ValueError("incomplete experiment")
    runs.sort(key=lambda x: x["protocol"]["seed"])
    seeds = [r["protocol"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed")
    first = runs[0]["protocol"]
    comparable = ("dataset", "outer_test_ratio", "inner_validation_ratio",
                  "fixed_clusters", "fixed_alpha", "neighbors", "top_k",
                  "validation_ranking_users_requested", "selection_objective")
    grid = [(c["clusters"], c["alpha"]) for c in runs[0]["selection"]["candidates"]]
    for run in runs:
        p = run["protocol"]
        if run["schema_version"] != 1 or any(p[k] != first[k] for k in comparable):
            raise ValueError("incomparable protocols")
        if [(c["clusters"], c["alpha"]) for c in run["selection"]["candidates"]] != grid:
            raise ValueError("incomparable candidate grids")
        if set(run["outer_test_results"]) != set(VARIANTS):
            raise ValueError("missing baseline")
        if p["outer_ranking_users_evaluated"] < 1:
            raise ValueError("empty ranking population")
        reference = run["outer_test_results"]["fixed_kurcf"]
        for row in run["outer_test_results"].values():
            if (row["n"] != p["outer_test_interactions"] or
                row["users"] != reference["users"] or
                row["positives"] != reference["positives"]):
                raise ValueError("mismatched evaluation populations")
    return {
        "schema_version": 1,
        "protocol": {
            "dataset": first["dataset"], "seeds": seeds,
            "validation_ranking_users_requested":
                first["validation_ranking_users_requested"],
            "validation_ranking_users_evaluated":
                [r["protocol"]["validation_ranking_users_evaluated"] for r in runs],
            "outer_ranking_users_evaluated":
                [r["protocol"]["outer_ranking_users_evaluated"] for r in runs],
            "outer_test_interactions":
                [r["protocol"]["outer_test_interactions"] for r in runs],
            "candidate_grid": [{"clusters": k, "alpha": a} for k, a in grid],
            "fingerprints": [{
                "seed": r["protocol"]["seed"],
                "inner_train": r["protocol"]["inner_training_sha256"],
                "inner_val": r["protocol"]["inner_validation_sha256"],
                "outer_train": r["protocol"]["outer_training_sha256"],
                "outer_test": r["protocol"]["outer_test_sha256"],
            } for r in runs],
            "ci_method": "exploratory 95% seed bootstrap, 4000 draws",
            "caveat": "Five overlapping splits of one dataset; not independent datasets.",
        },
        "selections": [{
            "seed": r["protocol"]["seed"],
            "mae": r["selection"]["rating_mae"],
            "recall": r["selection"]["ranking_recall"],
            "pareto_count": len(r["selection"]["pareto"]),
        } for r in runs],
        "metrics": {
            name: {
                metric: stats([float(r["outer_test_results"][name][metric]) for r in runs])
                for metric in METRICS
            } for name in VARIANTS
        },
        "paired_baseline_minus_recall_selected": {
            name: {
                metric: stats([float(r["outer_test_results"][name][metric]) -
                               float(r["outer_test_results"]["recall_selected"][metric])
                               for r in runs])
                for metric in METRICS
            } for name in VARIANTS if name != "recall_selected"
        },
    }


def make_svg(summary: dict) -> str:
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" '
             'viewBox="0 0 1280 760" role="img" aria-label="MovieLens multiobjective results">',
             '<title>MovieLens 100K: validation objective versus outer test</title>',
             '<rect width="1280" height="760" fill="#ffffff"/>']
    def label(x, y, t, size=16, color="#283749"):
        lines.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" '
                     f'font-size="{size}" fill="{color}">{escape(str(t))}</text>')
    label(42, 55, "KUR-CF | Validation objective versus outer-test quality", 29)
    label(42, 86, "Five real splits; error bars: exploratory 95% bootstrap interval", 16, "#60748a")
    for col, metric in enumerate(("mae", "recall_at_k")):
        x0 = 42 + col * 624
        lines.append(f'<rect x="{x0}" y="122" width="600" height="566" '
                     'rx="15" fill="#f8fafc" stroke="#e0e8f0"/>')
        label(x0 + 22, 166, "MAE (lower better)" if col == 0 else "Recall@10 (%) (higher better)", 21)
        scale = 100 if col else 1
        maximum = max(summary["metrics"][v][metric]["ci95"][1] * scale
                      for v in VARIANTS) * 1.2
        maximum = max(maximum, .00001)
        for idx, name in enumerate(VARIANTS):
            y = 207 + idx * 89
            row = summary["metrics"][name][metric]
            mean = row["mean"] * scale
            lo, hi = [v * scale for v in row["ci95"]]
            label(x0 + 24, y, LABELS[idx], 15)
            x = x0 + 24
            width = 445
            lines.append(f'<rect x="{x}" y="{y+12}" width="{width}" height="28" rx="5" fill="#e8edf3"/>')
            lines.append(f'<rect x="{x}" y="{y+12}" width="{width*mean/maximum:.2f}" '
                         f'height="28" rx="5" fill="{COLORS[idx]}"/>')
            lines.append(f'<line x1="{x+width*lo/maximum:.2f}" y1="{y+26}" '
                         f'x2="{x+width*hi/maximum:.2f}" y2="{y+26}" '
                         'stroke="#283749" stroke-width="2"/>')
            label(x0 + 483, y + 32, f"{mean:.4f}", 16)
    label(42, 732, "Hyperparameters selected on inner validation; outer-test labels never used for tuning.", 16, "#60748a")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def make_report(summary: dict) -> str:
    p = summary["protocol"]
    lines = [
        "# Multiobjective MovieLens 100K experiment",
        "",
        "**Real measurements only.** Both candidate objectives were selected on the",
        "inner validation split. Models were then refit and evaluated on the",
        "same untouched outer test. Validation ranking used a prespecified",
        f"sample of {p['validation_ranking_users_requested']} eligible users per split;",
        "outer-test ranking evaluated **all eligible users**.",
        "",
        "![Multiobjective comparison](../../assets/figures/ml100k-multiobjective.svg)",
        "",
        "## Outer-test metrics: mean ± sample SD across five splits",
        "",
        "| Model | MAE ↓ | RMSE ↓ | Precision@10 (%) ↑ | Recall@10 (%) ↑ |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name, label in zip(VARIANTS, LABELS):
        values = []
        for metric in METRICS:
            row = summary["metrics"][name][metric]
            scale = 100 if metric.endswith("_at_k") else 1
            values.append(f"{row['mean']*scale:.4f} ± {row['sd']*scale:.4f}")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines.extend(["", "## Validation-only selections", "",
                  "| Seed | MAE-selected K | MAE-selected alpha | Recall-selected K | Recall-selected alpha | Pareto candidates |",
                  "|--:|--:|--:|--:|--:|--:|"])
    for row in summary["selections"]:
        lines.append(f"| {row['seed']} | {row['mae']['clusters']} | {row['mae']['alpha']:.2f} | "
                     f"{row['recall']['clusters']} | {row['recall']['alpha']:.2f} | "
                     f"{row['pareto_count']} |")
    lines.extend([
        "", "## Interpretation constraints", "",
        "- The validation ranking objective uses a sampled population; test ranking is full-user.",
        "- Candidate grids were prespecified, not tuned using the outer test.",
        "- Five overlapping partitions from one dataset do not imply independent replication.",
        "- Original paper partitions and complete prediction equation remain unavailable.",
        "- [Raw observations](runs/) and [machine-readable summary](ml100k-summary.json).",
        "", "## Reproducibility", "", "~~~bash",
        "python scripts/summarize_multiobjective.py",
        "python scripts/summarize_multiobjective.py --check",
        "~~~", "",
    ])
    return "\n".join(lines)


def generate(directory: Path, min_seeds: int = 5) -> dict[str, str]:
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(directory.glob("seed-*.json"))]
    summary = aggregate(runs, min_seeds)
    return {
        "benchmarks/multiobjective/ml100k-summary.json":
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "benchmarks/multiobjective/README.md": make_report(summary),
        "assets/figures/ml100k-multiobjective.svg": make_svg(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("benchmarks/multiobjective/runs"))
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in generate(args.runs_dir, args.min_seeds).items():
        target = Path(path)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale multiobjective artifact: {path}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        print("Verified" if args.check else "Wrote", path)


if __name__ == "__main__":
    main()
