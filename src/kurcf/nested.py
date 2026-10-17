"""Leakage-free inner-validation selection and outer-test KUR-CF evaluation.

Independent protocol: the original paper's partitions and prediction rules
were not fully published.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .core import KURCF
from .data import load_movielens, user_stratified_split
from .evaluation import rating_metrics, ranking_metrics
from .study import _fingerprint


def choose_candidate(candidates: list[dict]) -> dict:
    """Validation MAE, then RMSE, then smaller K and alpha. No test metrics."""
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    for row in candidates:
        if any(not math.isfinite(float(row[key])) for key in ("mae", "rmse", "alpha")):
            raise ValueError("validation metrics and alpha must be finite")
        if row["clusters"] < 1 or not 0 <= row["alpha"] <= 1:
            raise ValueError("invalid candidate hyperparameters")
    return min(candidates, key=lambda r: (
        r["mae"], r["rmse"], r["clusters"], r["alpha"]
    )).copy()


def run_nested(args: Any) -> dict:
    """Tune on inner validation only, refit on outer train, test once."""
    if not 0 < args.test_ratio < 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("split ratios must be in (0,1)")
    if args.ranking_users < 0 or args.neighbors < 1 or args.top_k < 1:
        raise ValueError("ranking_users >= 0, neighbors >= 1, top_k >= 1 required")
    if not args.alpha_grid or not args.cluster_grid:
        raise ValueError("alpha and cluster grids cannot be empty")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alpha_grid):
        raise ValueError("alpha values must be finite in [0,1]")
    if any(not isinstance(k, int) or k < 2 for k in args.cluster_grid):
        raise ValueError("cluster counts must be integers >= 2")
    if not 0 <= args.fixed_alpha <= 1 or args.fixed_clusters < 2:
        raise ValueError("fixed model parameters are invalid")

    ratings, features = load_movielens(args.dataset, args.data_dir)
    outer_train, outer_test = user_stratified_split(ratings, args.test_ratio, args.seed)
    inner_train, inner_val = user_stratified_split(
        outer_train, args.validation_ratio, args.seed + 100_000
    )
    if not inner_val or not outer_test:
        raise ValueError("the dataset is too small for nested holdout")
    n_users = len({user for user, _, _ in inner_train})
    if max(args.cluster_grid + [args.fixed_clusters]) > n_users:
        raise ValueError("cluster count exceeds the number of training users")
    train_keys = {(u, i) for u, i, _ in inner_train}
    val_keys = {(u, i) for u, i, _ in inner_val}
    test_keys = {(u, i) for u, i, _ in outer_test}
    if (train_keys & val_keys) or (train_keys & test_keys) or (val_keys & test_keys):
        raise ValueError("train, validation and test partitions overlap")

    candidates: list[dict] = []
    for clusters in sorted(set(args.cluster_grid)):
        for alpha in sorted(set(args.alpha_grid)):
            model = KURCF(n_clusters=clusters, n_neighbors=args.neighbors,
                          alpha=alpha, seed=args.seed)
            start = time.perf_counter()
            model.fit(inner_train, features)
            score = rating_metrics(model, inner_val)
            candidates.append({
                "clusters": clusters, "alpha": alpha,
                "mae": score["mae"], "rmse": score["rmse"],
                "n": score["n"],
                "fit_and_validation_seconds": round(time.perf_counter() - start, 5),
            })
    best = choose_candidate(candidates)

    def evaluate(*, clustered: bool = True, similarity: str = "combined",
                 clusters: int | None = None, alpha: float | None = None) -> dict:
        model = KURCF(
            n_clusters=args.fixed_clusters if clusters is None else clusters,
            n_neighbors=args.neighbors,
            alpha=args.fixed_alpha if alpha is None else alpha,
            seed=args.seed, clustered=clustered, similarity=similarity,
        )
        start = time.perf_counter()
        model.fit(outer_train, features)
        fit_time = time.perf_counter() - start
        rating = rating_metrics(model, outer_test)
        ranking = ranking_metrics(
            model, outer_test, top_k=args.top_k,
            max_users=None if args.ranking_users == 0 else args.ranking_users,
            seed=args.seed,
        )
        return {
            **rating, **ranking,
            "fit_seconds": round(fit_time, 5),
            "total_seconds": round(time.perf_counter() - start, 5),
        }

    results = {
        "fixed_kurcf": evaluate(),
        "selected_kurcf": evaluate(clusters=best["clusters"], alpha=best["alpha"]),
        "cluster_jaccard": evaluate(similarity="jaccard"),
        "unclustered_hybrid": evaluate(clustered=False),
        "usercf": evaluate(clustered=False, similarity="pearson"),
    }
    expected = results["fixed_kurcf"]
    if any(result["n"] != len(outer_test) or
           result["users"] != expected["users"] or
           result["positives"] != expected["positives"]
           for result in results.values()):
        raise RuntimeError("evaluation populations differ between methods")

    return {
        "schema_version": 1,
        "protocol": {
            "dataset": args.dataset, "seed": args.seed,
            "outer_test_ratio": args.test_ratio,
            "inner_validation_ratio": args.validation_ratio,
            "inner_training_interactions": len(inner_train),
            "inner_validation_interactions": len(inner_val),
            "outer_training_interactions": len(outer_train),
            "outer_test_interactions": len(outer_test),
            "inner_training_sha256": _fingerprint(inner_train),
            "inner_validation_sha256": _fingerprint(inner_val),
            "outer_training_sha256": _fingerprint(outer_train),
            "outer_test_sha256": _fingerprint(outer_test),
            "fixed_clusters": args.fixed_clusters,
            "fixed_alpha": args.fixed_alpha,
            "neighbors": args.neighbors,
            "top_k": args.top_k,
            "ranking_users_requested": args.ranking_users,
            "ranking_users_evaluated": expected["users"],
            "ranking_population": "all eligible held-out-positive users" if args.ranking_users == 0
                                  else "sampled eligible held-out-positive users",
            "ranking_candidates": "all outer-training-unseen catalog items",
            "relevant_threshold": 4.0,
            "selection_objective": "full inner-validation MAE; no outer-test tuning",
            "paper_split_reproduced": False,
        },
        "selection": {
            "selected": {"clusters": best["clusters"], "alpha": best["alpha"]},
            "validation_mae": best["mae"],
            "validation_rmse": best["rmse"],
            "tie_break": "lowest validation MAE, then RMSE, then K, then alpha",
            "candidates": candidates,
        },
        "outer_test_results": results,
    }
