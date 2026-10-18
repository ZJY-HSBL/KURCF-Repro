"""Leakage-free rating-vs-ranking multiobjective study on MovieLens.

Every candidate is selected using the inner validation partition only.
Outer test is touched only after both selection objectives are fixed.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .core import KURCF
from .data import load_movielens, user_stratified_split
from .evaluation import rating_metrics, ranking_metrics
from .objectives import select_objectives
from .study import _fingerprint


def run_multiobjective(args: Any) -> dict:
    if not 0 < args.test_ratio < 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("invalid split ratio")
    if not args.cluster_grid or not args.alpha_grid:
        raise ValueError("empty candidate grid")
    if any(not isinstance(k, int) or k < 2 for k in args.cluster_grid):
        raise ValueError("invalid cluster grid")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alpha_grid):
        raise ValueError("invalid alpha grid")
    if args.neighbors < 1 or args.top_k < 1 or args.validation_ranking_users < 0:
        raise ValueError("invalid ranking configuration")
    if args.fixed_clusters < 2 or not 0 <= args.fixed_alpha <= 1:
        raise ValueError("invalid fixed baseline")

    ratings, features = load_movielens(args.dataset, args.data_dir)
    outer_train, outer_test = user_stratified_split(ratings, args.test_ratio, args.seed)
    inner_train, inner_val = user_stratified_split(
        outer_train, args.validation_ratio, args.seed + 100_000
    )
    if not inner_val or not outer_test:
        raise ValueError("empty holdout")
    if max([args.fixed_clusters, *args.cluster_grid]) > len({u for u, _, _ in inner_train}):
        raise ValueError("too many clusters for training users")
    keys = [{(u, i) for u, i, _ in split} for split in
            (inner_train, inner_val, outer_test)]
    if any(keys[i] & keys[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("overlapping partitions")

    candidates = []
    validation_population = None
    for k in sorted(set(args.cluster_grid)):
        for alpha in sorted(set(args.alpha_grid)):
            model = KURCF(n_clusters=k, n_neighbors=args.neighbors,
                          alpha=alpha, seed=args.seed)
            start = time.perf_counter()
            model.fit(inner_train, features)
            rating = rating_metrics(model, inner_val)
            ranking = ranking_metrics(
                model, inner_val, top_k=args.top_k,
                max_users=None if args.validation_ranking_users == 0
                          else args.validation_ranking_users,
                seed=args.seed,
            )
            population = (ranking["users"], ranking["positives"])
            if validation_population is None:
                validation_population = population
            elif population != validation_population:
                raise RuntimeError("validation ranking populations differ")
            candidates.append({
                "clusters": k, "alpha": alpha,
                "mae": rating["mae"], "rmse": rating["rmse"],
                "precision_at_k": ranking["precision_at_k"],
                "recall_at_k": ranking["recall_at_k"],
                "validation_ratings": rating["n"],
                "validation_ranking_users": ranking["users"],
                "validation_positives": ranking["positives"],
                "validation_seconds": round(time.perf_counter() - start, 5),
            })
    if validation_population is None or validation_population[0] == 0:
        raise ValueError("no eligible validation ranking users")
    selection = select_objectives(candidates)

    def outer_evaluate(k: int, alpha: float, *,
                       clustered: bool = True, similarity: str = "combined") -> dict:
        model = KURCF(n_clusters=k, n_neighbors=args.neighbors, alpha=alpha,
                      seed=args.seed, clustered=clustered, similarity=similarity)
        start = time.perf_counter()
        model.fit(outer_train, features)
        fit_seconds = time.perf_counter() - start
        rating = rating_metrics(model, outer_test)
        ranking = ranking_metrics(model, outer_test, top_k=args.top_k,
                                  max_users=None, seed=args.seed)
        return {**rating, **ranking, "fit_seconds": round(fit_seconds, 5),
                "total_seconds": round(time.perf_counter() - start, 5)}

    rating_best = selection["rating_mae"]
    ranking_best = selection["ranking_recall"]
    results = {
        "fixed_kurcf": outer_evaluate(args.fixed_clusters, args.fixed_alpha),
        "mae_selected": outer_evaluate(rating_best["clusters"], rating_best["alpha"]),
        "recall_selected": outer_evaluate(ranking_best["clusters"], ranking_best["alpha"]),
        "unclustered_hybrid": outer_evaluate(
            args.fixed_clusters, args.fixed_alpha, clustered=False),
        "usercf": outer_evaluate(args.fixed_clusters, args.fixed_alpha,
                                 clustered=False, similarity="pearson"),
    }
    reference = results["fixed_kurcf"]
    if any(r["n"] != len(outer_test) or r["users"] != reference["users"] or
           r["positives"] != reference["positives"] for r in results.values()):
        raise RuntimeError("outer-test evaluation populations differ")

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
            "neighbors": args.neighbors, "top_k": args.top_k,
            "validation_ranking_users_requested": args.validation_ranking_users,
            "validation_ranking_users_evaluated": validation_population[0],
            "outer_ranking_users_evaluated": reference["users"],
            "ranking_candidates": "all training-unseen catalog items",
            "relevant_threshold": 4.0,
            "selection_objective": "inner validation MAE vs Recall@K; never outer test",
            "paper_split_reproduced": False,
        },
        "selection": {
            "rating_mae": {"clusters": rating_best["clusters"],
                           "alpha": rating_best["alpha"]},
            "ranking_recall": {"clusters": ranking_best["clusters"],
                               "alpha": ranking_best["alpha"]},
            "pareto": [{"clusters": r["clusters"], "alpha": r["alpha"],
                        "mae": r["mae"], "recall_at_k": r["recall_at_k"]}
                       for r in selection["pareto"]],
            "candidates": candidates,
        },
        "outer_test_results": results,
    }
