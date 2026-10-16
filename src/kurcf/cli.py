"""Command line experiments for independent KUR-CF reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .core import KURCF, weighted_kmeans
from .data import download_movielens, load_movielens, user_stratified_split
from .evaluation import rating_metrics, ranking_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kurcf", description="Scientific Reports KUR-CF reproduction")
    subs = p.add_subparsers(dest="command", required=True)
    dl = subs.add_parser("download")
    dl.add_argument("--dataset", choices=["ml-100k", "ml-1m"], required=True)
    dl.add_argument("--data-dir", default="data")
    for command in ("benchmark", "sweep-k"):
        arg = subs.add_parser(command)
        arg.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
        arg.add_argument("--data-dir", default="data")
        arg.add_argument("--seed", type=int, default=42)
        arg.add_argument("--output", type=Path)
        if command == "benchmark":
            arg.add_argument("--models", nargs="+",
                             choices=["kurcf", "usercf", "cluster-pearson"],
                             default=["kurcf", "usercf", "cluster-pearson"])
            arg.add_argument("--clusters", type=int)
            arg.add_argument("--neighbors", type=int, default=20)
            arg.add_argument("--alpha", type=float, default=0.3)
            arg.add_argument("--test-ratio", type=float, default=0.2)
            arg.add_argument("--top-k", type=int, default=10)
            arg.add_argument("--ranking-users", type=int, default=100)
            arg.add_argument("--rating-cases", type=int, default=3000)
        else:
            arg.add_argument("--min-k", type=int, default=4)
            arg.add_argument("--max-k", type=int, default=12)
            arg.add_argument("--sample-size", type=int, default=1000)
    return p


def benchmark(args: argparse.Namespace) -> dict:
    ratings, features = load_movielens(args.dataset, args.data_dir)
    train, test = user_stratified_split(ratings, args.test_ratio, args.seed)
    clusters = args.clusters or (7 if args.dataset == "ml-100k" else 6)
    output = {
        "configuration": {
            "dataset": args.dataset, "seed": args.seed, "train_interactions": len(train),
            "test_interactions": len(test), "clusters": clusters, "neighbors": args.neighbors,
            "alpha": args.alpha, "top_k": args.top_k, "rating_cases": args.rating_cases,
            "ranking_users": args.ranking_users,
            "note": "Independent randomized split; does not claim identical paper partitions.",
        },
        "results": {},
    }
    for name in args.models:
        model = KURCF(
            n_clusters=clusters, n_neighbors=args.neighbors, alpha=args.alpha,
            seed=args.seed, clustered=name != "usercf",
            similarity="combined" if name == "kurcf" else "pearson"
        )
        start = time.perf_counter()
        model.fit(train, features)
        fit_time = time.perf_counter() - start
        rating = rating_metrics(model, test, max_cases=args.rating_cases, seed=args.seed)
        ranking = ranking_metrics(model, test, top_k=args.top_k,
                                  max_users=args.ranking_users, seed=args.seed)
        output["results"][name] = {
            **rating, **ranking, "fit_seconds": round(fit_time, 4),
            "total_seconds": round(time.perf_counter() - start, 4),
        }
    return output


def sweep_k(args: argparse.Namespace) -> dict:
    try:
        import numpy as np
        from sklearn.metrics import silhouette_score
    except ImportError as error:
        raise SystemExit("Install analysis dependencies: pip install -e '.[analysis]'") from error
    ratings, features = load_movielens(args.dataset, args.data_dir)
    train, _ = user_stratified_split(ratings, seed=args.seed)
    fitted = KURCF(n_clusters=2, seed=args.seed).fit(train, features)
    x = fitted.user_profiles
    results = {}
    for k in range(args.min_k, min(args.max_k, len(x) - 1) + 1):
        clusters = weighted_kmeans(x, k, seed=args.seed)
        scaled = x * np.sqrt(clusters.weights)
        if len(set(clusters.labels)) < 2:
            continue
        results[str(k)] = float(silhouette_score(
            scaled, clusters.labels, sample_size=min(len(x), args.sample_size),
            random_state=args.seed,
        ))
    return {
        "dataset": args.dataset, "silhouette": results,
        "best_k": int(max(results, key=results.get)) if results else None,
        "note": "Sampled conventional silhouette; paper's wording for silhouette differs.",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        print(download_movielens(args.dataset, args.data_dir))
        return
    result = benchmark(args) if args.command == "benchmark" else sweep_k(args)
    body = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(body)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
