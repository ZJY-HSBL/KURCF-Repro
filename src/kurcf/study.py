"""Controlled multi-seed benchmark and ablation protocol.

This is an *independent* MovieLens protocol, not a numerical reconstruction
of the paper's unpublished train/test partitions. In particular, all
hyperparameter sweeps are descriptive evaluations on the held-out set,
NOT validation-set hyperparameter selection.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from .core import KURCF
from .data import load_movielens, user_stratified_split
from .evaluation import rating_metrics, ranking_metrics


def _fingerprint(rows: list[tuple[int, int, float]]) -> str:
    """Fingerprint ordered split rows to make provenance verifiable."""
    digest = hashlib.sha256()
    for user, item, rating in rows:
        digest.update(f"{user}:{item}:{rating:.6f}\n".encode("ascii"))
    return digest.hexdigest()


def run_study(args: Any) -> dict:
    """Run matched ablations on ONE user-stratified split.

    0 for rating_cases/ranking_users means full evaluation. All variants use
    the same test split, same sampled rating cases, same sampled ranking users
    and identical train-only metadata. Model initialization uses the split seed.
    """
    if args.rating_cases < 0 or args.ranking_users < 0:
        raise ValueError("rating_cases and ranking_users must be >= 0 (0 = all)")
    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1)")
    if args.clusters < 2 or args.neighbors < 1 or args.top_k < 1:
        raise ValueError("clusters >= 2, neighbors >= 1, top_k >= 1 required")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if not args.alpha_grid or not args.cluster_grid:
        raise ValueError("alpha_grid and cluster_grid cannot be empty")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alpha_grid):
        raise ValueError("alpha_grid values must be in [0,1]")
    if any(k < 2 for k in args.cluster_grid):
        raise ValueError("cluster_grid values must be >= 2")

    raw, features = load_movielens(args.dataset, args.data_dir)
    train, test = user_stratified_split(raw, args.test_ratio, args.seed)
    n_rating = None if args.rating_cases == 0 else args.rating_cases
    n_users = None if args.ranking_users == 0 else args.ranking_users
    results: dict[str, dict] = {}
    cache: dict[tuple, dict] = {}

    def evaluate(*, clustered=True, similarity="combined", alpha=None,
                 clusters=None, feature_weighting="variance",
                 projected_centers=True) -> dict:
        a = args.alpha if alpha is None else float(alpha)
        k = args.clusters if clusters is None else int(clusters)
        # Ignore unused clustering options when clustering is disabled.
        key = (clustered, similarity, a if similarity == "combined" else 0,
               k if clustered else 0, feature_weighting if clustered else "",
               projected_centers if clustered else False)
        if key in cache:
            return cache[key].copy()
        model = KURCF(
            n_clusters=k, n_neighbors=args.neighbors, alpha=a, seed=args.seed,
            clustered=clustered, similarity=similarity,
            projected_centers=projected_centers,
            feature_weighting=feature_weighting,
        )
        start = time.perf_counter()
        model.fit(train, features)
        fit_seconds = time.perf_counter() - start
        rating = rating_metrics(model, test, max_cases=n_rating, seed=args.seed)
        ranking = ranking_metrics(
            model, test, top_k=args.top_k, max_users=n_users, seed=args.seed
        )
        measured = {
            **rating, **ranking,
            "fit_seconds": round(fit_seconds, 5),
            "total_seconds": round(time.perf_counter() - start, 5),
        }
        cache[key] = measured
        return measured.copy()

    # Main comparisons: the first three match existing smoke baselines.
    results["kurcf"] = evaluate()
    results["usercf"] = evaluate(clustered=False, similarity="pearson")
    results["cluster_pearson"] = evaluate(similarity="pearson")
    results["cluster_jaccard"] = evaluate(similarity="jaccard")
    results["no_cluster_hybrid"] = evaluate(clustered=False)
    results["uniform_weights"] = evaluate(feature_weighting="uniform")
    results["mean_centers"] = evaluate(projected_centers=False)

    alpha_sweep = [
        {"alpha": float(a), **evaluate(alpha=float(a))}
        for a in sorted(set(args.alpha_grid + [args.alpha]))
    ]
    k_sweep = [
        {"clusters": int(k), **evaluate(clusters=int(k))}
        for k in sorted(set(args.cluster_grid + [args.clusters]))
    ]

    return {
        "schema_version": 1,
        "protocol": {
            "dataset": args.dataset,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "split": "user-stratified 80/20; at least one train rating per user",
            "train_interactions": len(train),
            "test_interactions": len(test),
            "train_sha256": _fingerprint(train),
            "test_sha256": _fingerprint(test),
            "clusters": args.clusters,
            "neighbors": args.neighbors,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "rating_cases_requested": args.rating_cases,
            "ranking_users_requested": args.ranking_users,
            "ranking_relevant_threshold": 4.0,
            "ranking_candidates": "all training-unseen catalog items",
            "rating_cases_evaluated": results["kurcf"]["n"],
            "ranking_users_evaluated": results["kurcf"]["users"],
            "hyperparameter_sweeps": "descriptive test-set sensitivity only; NOT tuning",
            "paper_split_reproduced": False,
        },
        "variants": results,
        "alpha_sweep": alpha_sweep,
        "cluster_sweep": k_sweep,
    }
