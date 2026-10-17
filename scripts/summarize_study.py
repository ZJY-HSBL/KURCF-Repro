#!/usr/bin/env python3
"""Summarize real multi-seed study JSON and render deterministic SVG charts.

All observations must come from kurcf study, never from generated or guessed
benchmarks. Bootstrap resampling is performed over independent split seeds.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import random
import statistics

METRICS = ("mae", "rmse", "precision_at_k", "recall_at_k")
VARIANTS = ("kurcf", "usercf", "cluster_pearson", "cluster_jaccard",
            "no_cluster_hybrid", "uniform_weights", "mean_centers")
LABELS = {
    "kurcf": "KUR-CF", "usercf": "UserCF", "cluster_pearson": "Cluster-Pearson",
    "cluster_jaccard": "Cluster-Jaccard", "no_cluster_hybrid": "Unclustered hybrid",
    "uniform_weights": "Uniform weights", "mean_centers": "Mean centers",
}
COLORS = ["#245eab", "#0d998c", "#d68a2e", "#9866b6",
          "#5b8cae", "#bf667a", "#65768a"]


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    a, b = math.floor(index), math.ceil(index)
    return sorted_values[a] + (index - a) * (sorted_values[b] - sorted_values[a])


def stats(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("non-finite or missing measurements")
    rng = random.Random(20261017)
    draws = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(4000)
    )
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95": [percentile(draws, .025), percentile(draws, .975)],
        "n_seeds": len(values),
    }


def aggregate(runs: list[dict], min_seeds: int = 5) -> dict:
    if len(runs) < min_seeds:
        raise ValueError(f"need {min_seeds} independent runs, got {len(runs)}")
    runs.sort(key=lambda x: x["protocol"]["seed"])
    seeds = [x["protocol"]["seed"] for x in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate split seed")
    ref = runs[0]["protocol"]
    fields = ("dataset", "test_ratio", "clusters", "neighbors", "alpha",
              "top_k", "rating_cases_requested", "ranking_users_requested",
              "ranking_candidates", "ranking_relevant_threshold")
    for run in runs:
        p = run["protocol"]
        if run["schema_version"] != 1 or any(p[k] != ref[k] for k in fields):
            raise ValueError("non-comparable study protocols")
        if set(run["variants"]) != set(VARIANTS):
            raise ValueError("missing ablation variant")
        if p["rating_cases_requested"] == 0 and (
            run["variants"]["kurcf"]["n"] != p["test_interactions"]
        ):
            raise ValueError("not all held-out rating cases were evaluated")
        if any(run["variants"][name]["n"] != run["variants"]["kurcf"]["n"] or
               run["variants"][name]["users"] != run["variants"]["kurcf"]["users"]
               for name in VARIANTS):
            raise ValueError("evaluation populations differ between variants")
        for key, value in (("alpha_sweep", "alpha"), ("cluster_sweep", "clusters")):
            if [row[value] for row in run[key]] != [row[value] for row in runs[0][key]]:
                raise ValueError("sweep grids differ between seeds")

    metrics = {
        name: {metric: stats([float(run["variants"][name][metric]) for run in runs])
               for metric in METRICS}
        for name in VARIANTS
    }
    timing = {
        name: {key: stats([float(run["variants"][name][key]) for run in runs])
               for key in ("fit_seconds", "total_seconds")}
        for name in VARIANTS
    }
    paired = {
        name: {metric: stats([
            float(run["variants"][name][metric]) -
            float(run["variants"]["kurcf"][metric]) for run in runs
        ]) for metric in METRICS}
        for name in VARIANTS[1:]
    }
    sweeps = {}
    for key, xkey in (("alpha_sweep", "alpha"), ("cluster_sweep", "clusters")):
        sweeps[key] = [
            {xkey: row[xkey],
             "metrics": {metric: stats([
                 float(next(v for v in run[key] if v[xkey] == row[xkey])[metric])
                 for run in runs
             ]) for metric in METRICS}}
            for row in runs[0][key]
        ]
    return {
        "schema_version": 1,
        "protocol": {
            "dataset": ref["dataset"], "seeds": seeds, "n_seeds": len(seeds),
            "train_interactions": [r["protocol"]["train_interactions"] for r in runs],
            "test_interactions": [r["protocol"]["test_interactions"] for r in runs],
            "rating_cases_evaluated": [r["protocol"]["rating_cases_evaluated"] for r in runs],
            "ranking_users_evaluated": [r["protocol"]["ranking_users_evaluated"] for r in runs],
            "clusters": ref["clusters"], "neighbors": ref["neighbors"],
            "alpha": ref["alpha"], "top_k": ref["top_k"],
            "ranking_users_requested": ref["ranking_users_requested"],
            "rating_cases_requested": ref["rating_cases_requested"],
            "split_sha256": [
                {"seed": r["protocol"]["seed"],
                 "train": r["protocol"]["train_sha256"],
                 "test": r["protocol"]["test_sha256"]}
                for r in runs
            ],
            "ci_method": "95% percentile bootstrap over split seeds; 4000 resamples",
            "note": "Independent protocol. Test-set sweeps are descriptive, NOT hyperparameter tuning.",
        },
        "metrics": metrics, "timing": timing,
        "paired_baseline_minus_kurcf": paired, **sweeps,
    }


def text(x: float, y: float, value: str, size: int = 15, *,
         color: str = "#26384f", bold: bool = False, anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')


def line(x1: float, y1: float, x2: float, y2: float, *,
         color: str = "#cbd5e1", weight: float = 1.0) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{color}" stroke-width="{weight}"/>')


def svg_start(title: str, width: int, height: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f'<title>{escape(title)}</title>',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>']


def overview_svg(summary: dict) -> str:
    p = summary["protocol"]
    out = svg_start("KUR-CF multi-seed baseline and ablation comparison", 1510, 1140)
    out.append(text(54, 58, "KUR-CF | Multi-seed benchmark", 32, bold=True))
    out.append(text(54, 93, f"MovieLens 100K · seeds {p['seeds']} · "
                    f"all {p['test_interactions'][0]:,} held-out ratings per seed · "
                    f"{p['ranking_users_requested']} ranking users per seed", 16, color="#64748b"))
    out.append(text(54, 121, "Whiskers = 95% bootstrap CI of the mean over split seeds; "
                    "exploratory independent protocol.", 14, color="#64748b"))
    for index, metric in enumerate(METRICS):
        x0 = 54 + index % 2 * 742
        y0 = 158 + index // 2 * 472
        out.append(f'<rect x="{x0}" y="{y0}" width="714" height="446" rx="17" '
                   'fill="#f8fafc" stroke="#e2e8f0"/>')
        multiplier = 100 if metric.endswith("_at_k") else 1
        title = ("Precision@10 (%)" if metric == "precision_at_k" else
                 "Recall@10 (%)" if metric == "recall_at_k" else metric.upper())
        out.append(text(x0 + 22, y0 + 43, title, 22, bold=True))
        upper = max(summary["metrics"][name][metric]["ci95"][1] * multiplier
                    for name in VARIANTS) * 1.22
        upper = max(upper, 1e-7)
        start, length = x0 + 212, 337
        for j, name in enumerate(VARIANTS):
            y = y0 + 79 + j * 49
            row = summary["metrics"][name][metric]
            value = row["mean"] * multiplier
            lo, hi = (v * multiplier for v in row["ci95"])
            out.append(text(x0 + 22, y + 19, LABELS[name], 14))
            out.append(f'<rect x="{start}" y="{y}" width="{length}" height="26" rx="5" fill="#e7edf4"/>')
            out.append(f'<rect x="{start}" y="{y}" width="{length*value/upper:.3f}" '
                       f'height="26" rx="5" fill="{COLORS[j]}"/>')
            left, right = start + length * max(lo, 0) / upper, start + length * hi / upper
            out.append(line(left, y + 13, right, y + 13, color="#213247", weight=2))
            out.append(line(left, y + 7, left, y + 19, color="#213247", weight=2))
            out.append(line(right, y + 7, right, y + 19, color="#213247", weight=2))
            out.append(text(start + length + 13, y + 19, f"{value:.4f}", 14, bold=True))
    out.append(text(54, 1124, "Data: benchmarks/study/runs/seed-*.json · "
                    "The same split and test sample are used for every variant within a seed",
                    13, color="#64748b"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def sensitivity_svg(summary: dict, key: str, xkey: str, title: str) -> str:
    rows = summary[key]
    out = svg_start(title, 1300, 660)
    out.append(text(52, 61, title, 30, bold=True))
    out.append(text(52, 92, "Means and 95% bootstrap CIs across five split seeds; "
                    "test-set sensitivity is NOT validation-set tuning.", 16, color="#64748b"))
    for panel, metric in enumerate(("mae", "recall_at_k")):
        x0 = 54 + panel * 638
        y0 = 137
        out.append(f'<rect x="{x0}" y="{y0}" width="608" height="477" rx="16" '
                   'fill="#f8fafc" stroke="#e2e8f0"/>')
        out.append(text(x0 + 25, y0 + 42,
                        "MAE (lower is better)" if metric == "mae"
                        else "Recall@10 (%) (higher is better)", 21, bold=True))
        scale = 100 if metric.endswith("_at_k") else 1
        lows = [row["metrics"][metric]["ci95"][0] * scale for row in rows]
        highs = [row["metrics"][metric]["ci95"][1] * scale for row in rows]
        margin = max((max(highs) - min(lows)) * .2, .005 if scale == 1 else .02)
        bottom, top = min(lows) - margin, max(highs) + margin
        left, right = x0 + 95, x0 + 558
        ytop, ybottom = y0 + 98, y0 + 393
        def xp(i: int) -> float:
            return left + (right - left) * i / max(1, len(rows) - 1)
        def yp(v: float) -> float:
            return ybottom - (v - bottom) / (top - bottom) * (ybottom - ytop)
        for tick in range(5):
            value = bottom + (top - bottom) * tick / 4
            y = yp(value)
            out.append(line(left, y, right, y, color="#dce4ed"))
            out.append(text(left - 9, y + 5, f"{value:.3f}", 12,
                            anchor="end", color="#64748b"))
        points = [(xp(i), yp(row["metrics"][metric]["mean"] * scale))
                  for i, row in enumerate(rows)]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            out.append(line(x1, y1, x2, y2, color="#245eab", weight=3))
        for i, row in enumerate(rows):
            x, y = points[i]
            lo, hi = [v * scale for v in row["metrics"][metric]["ci95"]]
            out.append(line(x, yp(lo), x, yp(hi), color="#0d998c", weight=2))
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#245eab"/>')
            out.append(text(x, ybottom + 30, str(row[xkey]), 15, anchor="middle"))
        out.append(text((left + right) / 2, ybottom + 70,
                        "Pearson mixture weight α" if xkey == "alpha"
                        else "Number of clusters K", 15,
                        anchor="middle", color="#64748b"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def markdown_report(summary: dict) -> str:
    p = summary["protocol"]
    rows = [
        "# MovieLens 100K — full rating evaluation and ablation study",
        "",
        "**Actual measured results** from [raw per-seed JSON](runs/), not paper figures.",
        "",
        f"Five independent stratified splits: seeds {p['seeds']}.",
        f"Rating evaluation covers the **entire held-out set** per seed: {p['rating_cases_evaluated']}.",
        f"Top-10 ranking covers **sampled users** per seed: {p['ranking_users_evaluated']}.",
        f"Settings: K={p['clusters']}, N={p['neighbors']}, α={p['alpha']}, relevant rating ≥ 4.",
        "95% bootstrap intervals are resampled over seeds (4,000 draws); five seeds are exploratory.",
        "",
        "![Baseline and ablation results](../../assets/figures/ml100k-study-overview.svg)",
        "",
        "## Mean ± sample standard deviation across seeds",
        "",
        "| Variant | MAE ↓ | RMSE ↓ | Precision@10 (%) ↑ | Recall@10 (%) ↑ |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name in VARIANTS:
        def cell(metric: str) -> str:
            s = summary["metrics"][name][metric]
            factor = 100 if metric.endswith("_at_k") else 1
            return f"{s['mean'] * factor:.4f} ± {s['sd'] * factor:.4f}"
        rows.append("| " + LABELS[name] + " | " +
                    " | ".join(cell(m) for m in METRICS) + " |")
    rows += [
        "",
        "[Full JSON summary](ml100k-summary.json) contains seed-level split "
        "fingerprints, 95% CIs, runtime and paired baseline-minus-KUR-CF differences.",
        "",
        "## Sensitivity sweeps",
        "",
        "![Alpha sensitivity](../../assets/figures/ml100k-study-alpha.svg)",
        "",
        "![Cluster-count sensitivity](../../assets/figures/ml100k-study-clusters.svg)",
        "",
        "The alpha and K sweeps use the held-out test set only for **descriptive analysis**. "
        "Do not select a best setting using these plots and report its test metric as unbiased; "
        "that requires a separate validation split.",
        "",
        "## Interpretation limits",
        "",
        "This is an independent protocol, not a strict numerical replication of the paper. "
        "The original data partitions and prediction details are not public in the paper. "
        "The full held-out **rating** set is evaluated, while Top-K ranking uses 100 "
        "sampled eligible users per seed. Paired multi-seed differences and CIs "
        "do not prove the paper's reported improvements.",
        "",
        "To regenerate figures and summary from committed raw observations:",
        "",
        "~~~bash",
        "python scripts/summarize_study.py --min-seeds 5",
        "python scripts/summarize_study.py --min-seeds 5 --check",
        "~~~",
        "",
    ]
    return "\n".join(rows)


def generate(runs_dir: Path, min_seeds: int = 5) -> dict[str, str]:
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(runs_dir.glob("seed-*.json"))]
    summary = aggregate(runs, min_seeds=min_seeds)
    return {
        "benchmarks/study/ml100k-summary.json":
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "benchmarks/study/README.md": markdown_report(summary),
        "assets/figures/ml100k-study-overview.svg": overview_svg(summary),
        "assets/figures/ml100k-study-alpha.svg":
            sensitivity_svg(summary, "alpha_sweep", "alpha", "KUR-CF | Alpha sensitivity"),
        "assets/figures/ml100k-study-clusters.svg":
            sensitivity_svg(summary, "cluster_sweep", "clusters",
                            "KUR-CF | Cluster-count sensitivity"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("benchmarks/study/runs"))
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in generate(args.runs_dir, min_seeds=args.min_seeds).items():
        path = Path(name)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Generated artifact is stale: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(("Verified" if args.check else "Wrote"), name)


if __name__ == "__main__":
    main()
