"""Leakage guards and deterministic nested-validation reporting."""
from argparse import Namespace
import copy
import math
import xml.etree.ElementTree as ET

import pytest

from kurcf.data import user_stratified_split
from kurcf.nested import choose_candidate, run_nested
from scripts.summarize_nested import aggregate, svg, markdown

ATTRS = {i: [float(i % 4 == j) for j in range(4)] for i in range(1, 13)}
RATINGS = [
    (u, i, float((u * 7 + i * 3) % 5 + 1))
    for u in range(1, 13) for i in range(1, 13)
    if (u + i) % 7 != 0
]


def args(seed=1):
    return Namespace(
        dataset="ml-100k", data_dir="unused", seed=seed,
        test_ratio=.2, validation_ratio=.2, ranking_users=0,
        neighbors=4, top_k=3, fixed_clusters=2, fixed_alpha=.3,
        alpha_grid=[0., .3, 1.], cluster_grid=[2, 3],
    )


def test_validation_selection_ties_and_bad_values():
    rows = [
        {"clusters": 3, "alpha": .3, "mae": .5, "rmse": .7},
        {"clusters": 2, "alpha": 1., "mae": .5, "rmse": .7},
        {"clusters": 2, "alpha": 0., "mae": .5, "rmse": .7},
    ]
    assert choose_candidate(rows)["alpha"] == 0
    with pytest.raises(ValueError):
        choose_candidate([])
    with pytest.raises(ValueError):
        choose_candidate([{**rows[0], "mae": math.nan}])


def test_full_outer_test_and_all_eligible_ranking(monkeypatch):
    monkeypatch.setattr("kurcf.nested.load_movielens", lambda *_: (RATINGS, ATTRS))
    run = run_nested(args())
    assert run["schema_version"] == 1
    assert len(run["selection"]["candidates"]) == 6
    assert set(run["outer_test_results"]) == {
        "fixed_kurcf", "selected_kurcf", "cluster_jaccard", "unclustered_hybrid", "usercf"
    }
    assert run["protocol"]["ranking_users_requested"] == 0
    assert run["protocol"]["ranking_users_evaluated"] > 0
    for row in run["outer_test_results"].values():
        assert row["n"] == run["protocol"]["outer_test_interactions"]
        assert row["users"] == run["protocol"]["ranking_users_evaluated"]
        assert math.isfinite(row["mae"])
    selected = run["selection"]["selected"]
    assert selected in [{"clusters": c["clusters"], "alpha": c["alpha"]}
                        for c in run["selection"]["candidates"]]


def test_outer_test_labels_cannot_change_selection(monkeypatch):
    options = args(seed=2)
    monkeypatch.setattr("kurcf.nested.load_movielens", lambda *_: (RATINGS, ATTRS))
    original = run_nested(options)
    _, outer_test = user_stratified_split(RATINGS, options.test_ratio, options.seed)
    held_out = {(u, i) for u, i, _ in outer_test}
    changed = [(u, i, float(6 - score) if (u, i) in held_out else score)
               for u, i, score in RATINGS]
    monkeypatch.setattr("kurcf.nested.load_movielens", lambda *_: (changed, ATTRS))
    modified = run_nested(options)
    assert original["selection"]["selected"] == modified["selection"]["selected"]
    for before, after in zip(original["selection"]["candidates"],
                             modified["selection"]["candidates"]):
        assert before["mae"] == pytest.approx(after["mae"])
        assert before["rmse"] == pytest.approx(after["rmse"])
    assert original["protocol"]["outer_test_sha256"] != modified["protocol"]["outer_test_sha256"]
    assert original["protocol"]["inner_validation_sha256"] == modified["protocol"]["inner_validation_sha256"]


def test_summary_validates_population_and_generates_svg(monkeypatch):
    monkeypatch.setattr("kurcf.nested.load_movielens", lambda *_: (RATINGS, ATTRS))
    one, two = run_nested(args(1)), run_nested(args(2))
    report = aggregate([copy.deepcopy(one), copy.deepcopy(two)], min_seeds=2)
    assert report["protocol"]["seeds"] == [1, 2]
    assert report["metrics"]["selected_kurcf"]["mae"]["n_seeds"] == 2
    assert sum(r["count"] for r in report["selection_counts"]) == 2
    ET.fromstring(svg(report))
    assert "outer test" in markdown(report).lower()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([one, one], min_seeds=2)
    modified = copy.deepcopy(two)
    modified["protocol"]["neighbors"] += 1
    with pytest.raises(ValueError, match="incomparable"):
        aggregate([one, modified], min_seeds=2)
    sampled = copy.deepcopy(two)
    sampled["protocol"]["ranking_users_requested"] = 2
    with pytest.raises(ValueError):
        aggregate([one, sampled], min_seeds=2)


def test_invalid_nested_args(monkeypatch):
    monkeypatch.setattr("kurcf.nested.load_movielens", lambda *_: (RATINGS, ATTRS))
    bad = args()
    bad.alpha_grid = [float("nan")]
    with pytest.raises(ValueError):
        run_nested(bad)
    bad = args()
    bad.cluster_grid = [200]
    with pytest.raises(ValueError):
        run_nested(bad)
