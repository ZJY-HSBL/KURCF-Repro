"""Multiobjective pipeline tests on synthetic explicit ratings."""
from argparse import Namespace
import math
import pytest

from kurcf.data import user_stratified_split
from kurcf.multiobjective import run_multiobjective

FEATURES = {i: [float(i % 4 == j) for j in range(4)] for i in range(1, 13)}
RATINGS = [(u, i, float((u * 7 + i * 3) % 5 + 1))
           for u in range(1, 13) for i in range(1, 13) if (u + i) % 7 != 0]


def settings():
    return Namespace(dataset="ml-100k", data_dir="unused", seed=1,
                     test_ratio=.2, validation_ratio=.2,
                     validation_ranking_users=0, neighbors=4, top_k=3,
                     fixed_clusters=2, fixed_alpha=.3,
                     alpha_grid=[0., .3], cluster_grid=[2, 3])


def test_both_objectives_use_inner_validation(monkeypatch):
    monkeypatch.setattr("kurcf.multiobjective.load_movielens",
                        lambda *_: (RATINGS, FEATURES))
    result = run_multiobjective(settings())
    assert len(result["selection"]["candidates"]) == 4
    assert set(result["outer_test_results"]) == {
        "fixed_kurcf", "mae_selected", "recall_selected",
        "unclustered_hybrid", "usercf"
    }
    assert result["protocol"]["validation_ranking_users_evaluated"] > 0
    assert result["protocol"]["outer_ranking_users_evaluated"] > 0
    for item in result["outer_test_results"].values():
        assert item["n"] == result["protocol"]["outer_test_interactions"]
        assert item["users"] == result["protocol"]["outer_ranking_users_evaluated"]
        assert math.isfinite(item["mae"])


def test_outer_test_label_mutation_cannot_change_selection(monkeypatch):
    config = settings()
    monkeypatch.setattr("kurcf.multiobjective.load_movielens",
                        lambda *_: (RATINGS, FEATURES))
    first = run_multiobjective(config)
    _, test = user_stratified_split(RATINGS, config.test_ratio, config.seed)
    test_keys = {(u, i) for u, i, _ in test}
    modified = [(u, i, float(6 - v) if (u, i) in test_keys else v)
                for u, i, v in RATINGS]
    monkeypatch.setattr("kurcf.multiobjective.load_movielens",
                        lambda *_: (modified, FEATURES))
    second = run_multiobjective(config)
    # Runtime measurements fluctuate between runs and are not selection criteria.
    def decision_fields(selection):
        return {
            **selection,
            "candidates": [
                {key: value for key, value in row.items() if key != "validation_seconds"}
                for row in selection["candidates"]
            ],
        }

    assert decision_fields(first["selection"]) == decision_fields(second["selection"])
    assert first["protocol"]["outer_test_sha256"] != second["protocol"]["outer_test_sha256"]


def test_rejects_invalid_configuration(monkeypatch):
    monkeypatch.setattr("kurcf.multiobjective.load_movielens",
                        lambda *_: (RATINGS, FEATURES))
    config = settings()
    config.alpha_grid = [float("nan")]
    with pytest.raises(ValueError):
        run_multiobjective(config)
