"""Validation-only multi-objective candidate selection for future KUR-CF studies.

Selection uses no outer-test scores. All objectives are explicit, with
deterministic tie breaks and a Pareto frontier for diagnostic reporting.
"""
from __future__ import annotations

import math
from typing import Iterable


def _check(rows: Iterable[dict]) -> list[dict]:
    candidates = list(rows)
    if not candidates:
        raise ValueError("empty validation candidate grid")
    required = ("clusters", "alpha", "mae", "rmse", "recall_at_k", "precision_at_k")
    for row in candidates:
        if any(key not in row for key in required):
            raise ValueError("missing validation metric or parameter")
        if any(not math.isfinite(float(row[k])) for k in required):
            raise ValueError("nonfinite validation metric or parameter")
        if int(row["clusters"]) != row["clusters"] or row["clusters"] < 2:
            raise ValueError("clusters must be an integer >=2")
        if not 0 <= row["alpha"] <= 1:
            raise ValueError("alpha outside [0,1]")
        if not 0 <= row["recall_at_k"] <= 1 or not 0 <= row["precision_at_k"] <= 1:
            raise ValueError("ranking metric outside [0,1]")
    return candidates


def pareto_frontier(rows: Iterable[dict]) -> list[dict]:
    """Nondominated on minimizing MAE and maximizing Recall@K."""
    candidates = _check(rows)
    frontier = []
    for row in candidates:
        dominated = any(
            (other["mae"] <= row["mae"] and
             other["recall_at_k"] >= row["recall_at_k"] and
             (other["mae"] < row["mae"] or
              other["recall_at_k"] > row["recall_at_k"]))
            for other in candidates
        )
        if not dominated:
            frontier.append(row.copy())
    return sorted(frontier, key=lambda x: (x["mae"], -x["recall_at_k"], x["clusters"], x["alpha"]))


def select_objectives(rows: Iterable[dict]) -> dict:
    """Return independently selected validation-only rating and ranking optima."""
    candidates = _check(rows)
    rating = min(candidates, key=lambda x: (
        x["mae"], x["rmse"], x["clusters"], x["alpha"]
    ))
    ranking = min(candidates, key=lambda x: (
        -x["recall_at_k"], -x["precision_at_k"], x["mae"],
        x["clusters"], x["alpha"]
    ))
    return {
        "rating_mae": rating.copy(),
        "ranking_recall": ranking.copy(),
        "pareto": pareto_frontier(candidates),
        "selection_data": "inner validation only",
    }
