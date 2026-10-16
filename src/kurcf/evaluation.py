"""Transparent explicit-rating and held-out Top-K evaluation."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable

from .core import KURCF
from .data import Rating


def rating_metrics(
    model: KURCF, test: Iterable[Rating], *, max_cases: int | None = None, seed: int = 42
) -> dict[str, float | int]:
    """MAE and RMSE on held-out interactions only."""
    rows = list(test)
    if max_cases is not None and len(rows) > max_cases:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        rows = random.Random(seed).sample(rows, max_cases)
    if not rows:
        return {"mae": math.nan, "rmse": math.nan, "n": 0}
    abs_err = 0.0
    squared = 0.0
    for user, item, actual in rows:
        delta = model.predict(user, item) - actual
        abs_err += abs(delta)
        squared += delta * delta
    return {"mae": abs_err / len(rows), "rmse": math.sqrt(squared / len(rows)), "n": len(rows)}


def ranking_metrics(
    model: KURCF, test: Iterable[Rating], *, top_k: int = 10,
    relevant_threshold: float = 4.0, max_users: int | None = 100, seed: int = 42,
) -> dict[str, float | int]:
    """Micro-averaged Precision@K and Recall@K for held-out relevant items.

    Includes only users with >=1 relevant item in test. Entire unseen catalog
    is eligible for each user's ranking. No test interactions enter fit().
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    relevant: dict[int, set[int]] = defaultdict(set)
    for user, item, value in test:
        if value >= relevant_threshold:
            relevant[user].add(item)
    users = sorted(u for u, positive in relevant.items() if positive and u in model.ratings)
    if max_users is not None and len(users) > max_users:
        if max_users < 1:
            raise ValueError("max_users must be positive")
        users = sorted(random.Random(seed).sample(users, max_users))
    hits = 0
    recommended = 0
    positives = 0
    for user in users:
        items = {movie for movie, _score in model.recommend(user, top_k)}
        hits += len(items & relevant[user])
        recommended += len(items)
        positives += len(relevant[user])
    return {
        "precision_at_k": hits / recommended if recommended else 0.0,
        "recall_at_k": hits / positives if positives else 0.0,
        "hits": hits, "recommended": recommended, "positives": positives,
        "users": len(users), "k": top_k,
    }
