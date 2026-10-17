#!/usr/bin/env python3
"""Summarize actual nested-validation seed JSON; no simulated observations."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

import math
import random
import statistics


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    a, b = math.floor(index), math.ceil(index)
    return values[a] + (index - a) * (values[b] - values[a])


def stats(values: list[float]) -> dict:
    """Sample SD and 95% seed-bootstrap CI; deterministic across invocations."""
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
        "ci95": [_percentile(draws, .025), _percentile(draws, .975)],
        "n_seeds": len(values),
    }

METRICS = ("mae", "rmse", "precision_at_k", "recall_at_k")
VARIANTS = ("fixed_kurcf", "selected_kurcf", "cluster_jaccard",
            "unclustered_hybrid", "usercf")
LABELS = {
    "fixed_kurcf": "KUR-CF (fixed)",
    "selected_kurcf": "KUR-CF (validation-selected)",
    "cluster_jaccard": "Cluster-Jaccard",
    "unclustered_hybrid": "Unclustered hybrid",
    "usercf": "UserCF (Pearson)",
}
COLORS = ("#527aaf", "#126f79", "#ba843b", "#9361aa", "#6d7e8b")


def aggregate(runs: list[dict], *, min_seeds: int = 5,
              require_full_ranking: bool = True) -> dict:
    if len(runs) < min_seeds:
        raise ValueError(f"expected at least {min_seeds} runs")
    runs.sort(key=lambda r: r["protocol"]["seed"])
    seeds = [r["protocol"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate random split seed")
    reference = runs[0]["protocol"]
    fields = ("dataset", "outer_test_ratio", "inner_validation_ratio",
              "fixed_clusters", "fixed_alpha", "neighbors", "top_k",
              "ranking_users_requested", "ranking_candidates",
              "relevant_threshold", "selection_objective")
    reference_grid = [(c["clusters"], c["alpha"])
                      for c in runs[0]["selection"]["candidates"]]
    for run in runs:
        p = run["protocol"]
        if run["schema_version"] != 1 or any(p[field] != reference[field] for field in fields):
            raise ValueError("incomparable nested evaluation protocols")
        if require_full_ranking and p["ranking_users_requested"] != 0:
            raise ValueError("published protocol requires full eligible-user ranking")
        if set(run["outer_test_results"]) != set(VARIANTS):
            raise ValueError("missing outer-test baseline")
        if [(c["clusters"], c["alpha"]) for c in run["selection"]["candidates"]] != reference_grid:
            raise ValueError("candidate grids differ between seeds")
        selected = run["selection"]["selected"]
        if (selected["clusters"], selected["alpha"]) not in reference_grid:
            raise ValueError("selected hyperparameters not in validation grid")
        for name in VARIANTS:
            row = run["outer_test_results"][name]
            if row["n"] != p["outer_test_interactions"]:
                raise ValueError("not all held-out ratings were evaluated")
            if row["users"] != p["ranking_users_evaluated"]:
                raise ValueError("ranking populations differ")
            if row["positives"] != run["outer_test_results"]["fixed_kurcf"]["positives"]:
                raise ValueError("relevance populations differ")

    metrics = {
        name: {
            metric: stats([float(run["outer_test_results"][name][metric]) for run in runs])
            for metric in METRICS
        }
        for name in VARIANTS
    }
    paired = {
        name: {
            metric: stats([
                float(run["outer_test_results"][name][metric]) -
                float(run["outer_test_results"]["selected_kurcf"][metric])
                for run in runs
            ])
            for metric in METRICS
        }
        for name in VARIANTS if name != "selected_kurcf"
    }
    choices = Counter((run["selection"]["selected"]["clusters"],
                       run["selection"]["selected"]["alpha"]) for run in runs)
    return {
        "schema_version": 1,
        "protocol": {
            "dataset": reference["dataset"], "seeds": seeds,
            "n_seeds": len(seeds),
            "outer_test_ratio": reference["outer_test_ratio"],
            "inner_validation_ratio": reference["inner_validation_ratio"],
            "ranking_users_requested": reference["ranking_users_requested"],
            "ranking_users_evaluated": [r["protocol"]["ranking_users_evaluated"] for r in runs],
            "outer_test_interactions": [r["protocol"]["outer_test_interactions"] for r in runs],
            "inner_validation_interactions": [r["protocol"]["inner_validation_interactions"] for r in runs],
            "split_fingerprints": [{
                "seed": r["protocol"]["seed"],
                "outer_train": r["protocol"]["outer_training_sha256"],
                "outer_test": r["protocol"]["outer_test_sha256"],
                "inner_train": r["protocol"]["inner_training_sha256"],
                "inner_val": r["protocol"]["inner_validation_sha256"],
            } for r in runs],
            "candidate_grid": [{"clusters": k, "alpha": a} for k, a in reference_grid],
            "selection_objective": reference["selection_objective"],
            "ci_method": "95% percentile bootstrap over five split seeds; 4000 draws",
            "caveat": "One dataset, overlapping split samples and few seeds; not independent dataset replication.",
        },
        "selection_counts": [
            {"clusters": k, "alpha": a, "count": count}
            for (k, a), count in sorted(choices.items())
        ],
        "metrics": metrics,
        "paired_baseline_minus_selected": paired,
    }


def svg(summary: dict) -> str:
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="770" '
        'viewBox="0 0 1260 770" role="img" aria-label="Nested validation test results">',
        '<title>MovieLens 100K — nested validation and outer test</title>',
        '<rect width="1260" height="770" fill="#ffffff"/>',
    ]
    def label(x, y, value, size=16, color="#243449", bold=False):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" '
                   f'font-size="{size}" font-weight="{700 if bold else 400}" '
                   f'fill="{color}">{escape(str(value))}</text>')
    label(48, 57, "KUR-CF | Nested validation vs. outer-test performance", 29, bold=True)
    label(48, 88, "Five seeds; full outer-test ratings and all eligible ranking users; "
          "error bars: 95% bootstrap CI", 16, "#62758b")
    for idx, metric in enumerate(("mae", "recall_at_k")):
        x0 = 48 + idx * 615
        y0 = 133
        out.append(f'<rect x="{x0}" y="{y0}" width="582" height="548" rx="16" '
                   'fill="#f8fafc" stroke="#e1e8f0"/>')
        title = "Outer-test MAE (lower is better)" if metric == "mae" else (
            "Outer-test Recall@10 (%) (higher is better)")
        label(x0 + 20, y0 + 44, title, 20, bold=True)
        factor = 100 if metric.endswith("_at_k") else 1
        upper = max(summary["metrics"][name][metric]["ci95"][1] * factor
                    for name in VARIANTS) * 1.16
        upper = max(upper, 1e-6)
        bar_start, bar_length = x0 + 24, 430
        for j, name in enumerate(VARIANTS):
            y = y0 + 95 + j * 85
            row = summary["metrics"][name][metric]
            mean = row["mean"] * factor
            lo, hi = (v * factor for v in row["ci95"])
            label(bar_start, y - 8, LABELS[name], 15)
            out.append(f'<rect x="{bar_start}" y="{y}" width="{bar_length}" '
                       'height="28" rx="5" fill="#e6edf4"/>')
            out.append(f'<rect x="{bar_start}" y="{y}" width="{bar_length * mean / upper:.3f}" '
                       f'height="28" rx="5" fill="{COLORS[j]}"/>')
            a = bar_start + bar_length * max(lo, 0) / upper
            b = bar_start + bar_length * hi / upper
            out.append(f'<line x1="{a:.2f}" y1="{y+14}" x2="{b:.2f}" y2="{y+14}" '
                       'stroke="#23354a" stroke-width="2"/>')
            label(bar_start + bar_length + 12, y + 20, f"{mean:.4f}", 14, bold=True)
    label(48, 735, "Validation selects hyperparameters without outer-test labels. "
          "Original paper partitions remain unknown.", 15, "#65778b")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def markdown(summary: dict) -> str:
    p = summary["protocol"]
    lines = [
        "# MovieLens 100K — nested validation and full eligible-user ranking",
        "",
        "**Actual measurements from [raw seed JSON](runs/).** Hyperparameters are "
        "selected on an **inner validation split**; the selected model is then "
        "refit on outer training and evaluated on the held-out outer test.",
        "",
        f"- Outer 80/20 seeds: {p['seeds']}.",
        f"- Inner validation fraction of outer training: {p['inner_validation_ratio']:.0%}.",
        f"- Outer test ratings per seed: {p['outer_test_interactions']} (all evaluated).",
        f"- Eligible ranking users per seed: {p['ranking_users_evaluated']} "
        "(all with held-out rating >=4, not sampled).",
        "- All baselines use the **same outer train/test split** and ranking population.",
        "- Selection: lowest inner-validation MAE; tie-break RMSE, K, alpha.",
        "- 95% bootstrap intervals resample five split seeds, not individual ratings.",
        "",
        "![Nested validation and outer-test results](../../assets/figures/ml100k-nested-overview.svg)",
        "",
        "## Outer-test mean ± sample SD across five seeds",
        "",
        "| Model | MAE ↓ | RMSE ↓ | Precision@10 (%) ↑ | Recall@10 (%) ↑ |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name in VARIANTS:
        def cell(metric):
            s = summary["metrics"][name][metric]
            scale = 100 if metric.endswith("_at_k") else 1
            return f"{s['mean'] * scale:.4f} ± {s['sd'] * scale:.4f}"
        lines.append("| " + LABELS[name] + " | " + " | ".join(cell(m) for m in METRICS) + " |")
    lines.extend([
        "", "## Inner-validation selections (not test-selected)", "",
        "| Selected K | Selected alpha | Seeds |", "|--:|--:|--:|",
    ])
    for row in summary["selection_counts"]:
        lines.append(f"| {row['clusters']} | {row['alpha']:.2f} | {row['count']} |")
    lines.extend([
        "", "## Paired outer-test differences", "",
        "**Baseline minus validation-selected KUR-CF**, paired by outer split. "
        "Positive MAE/RMSE differences favor selected KUR-CF; positive "
        "Precision/Recall differences favor the baseline.",
        "",
        "| Baseline | ΔMAE | ΔRMSE | ΔPrecision@10 (pp) | ΔRecall@10 (pp) |",
        "|:--|--:|--:|--:|--:|",
    ])
    for name in VARIANTS:
        if name == "selected_kurcf":
            continue
        values = summary["paired_baseline_minus_selected"][name]
        lines.append("| " + LABELS[name] + " | " + " | ".join(
            f"{values[metric]['mean'] * (100 if metric.endswith('_at_k') else 1):+.4f}"
            for metric in METRICS
        ) + " |")
    lines.extend([
        "", "[Machine-readable summary](ml100k-summary.json) includes exploratory "
        "95% intervals, paired differences and SHA-256 partition fingerprints.",
        "", "## Limitations", "",
        "- Hyperparameter selection optimizes **rating MAE**, not ranking quality.",
        "- Inner and outer partitions share users, but not interactions. "
        "No test ratings enter fitting or selection.",
        "- Five partitions of **one dataset** are not five independent datasets.",
        "- Original paper partitions and full prediction equation are unavailable.",
        "", "## Rebuild", "", "~~~bash",
        "python scripts/summarize_nested.py --min-seeds 5",
        "python scripts/summarize_nested.py --min-seeds 5 --check",
        "~~~", "",
    ])
    return "\n".join(lines)


def generate(directory: Path, min_seeds: int = 5) -> dict[str, str]:
    runs = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("seed-*.json"))]
    summary = aggregate(runs, min_seeds=min_seeds)
    return {
        "benchmarks/nested/ml100k-summary.json":
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "benchmarks/nested/README.md": markdown(summary),
        "assets/figures/ml100k-nested-overview.svg": svg(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("benchmarks/nested/runs"))
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in generate(args.runs_dir, min_seeds=args.min_seeds).items():
        target = Path(name)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale nested-study artifact: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        print(("Verified" if args.check else "Wrote"), name)


if __name__ == "__main__":
    main()
