"""Equation-level checks and no-leakage integration tests."""
import math
import numpy as np
import pytest
from kurcf.core import (
    KURCF, variance_weights, weighted_kmeans,
    pearson_similarity, jaccard_similarity, combined_similarity,
)
from kurcf.data import user_stratified_split, load_movielens, download_movielens
from kurcf.evaluation import rating_metrics, ranking_metrics
from kurcf.cli import build_parser

MOVIES = {
    1: [1, 0], 2: [1, 0], 3: [0, 1],
    4: [0, 1], 5: [1, 1], 6: [0, 1],
}
TRAIN = [
    (1, 1, 5), (1, 2, 4), (2, 1, 4), (2, 2, 5), (2, 3, 1),
    (3, 3, 5), (3, 4, 4), (4, 3, 4), (4, 4, 5), (4, 5, 4),
]
TEST = [(1, 3, 2), (3, 2, 1), (4, 6, 5)]


def test_weights():
    np.testing.assert_allclose(
        variance_weights(np.array([[0, 1], [2, 1], [4, 1]])), [1, 0]
    )
    np.testing.assert_allclose(variance_weights(np.ones((4, 3))), [1/3] * 3)
    with pytest.raises(ValueError):
        variance_weights(np.array([[float("nan")]]))


def test_cluster_stable():
    x = np.array([[0, 0], [0, 0], [0.1, 0], [10, 10], [10, 10], [10.1, 10]])
    one = weighted_kmeans(x, 2, seed=7)
    two = weighted_kmeans(x, 2, seed=7)
    np.testing.assert_array_equal(one.labels, two.labels)
    assert len(np.unique(one.labels)) == 2
    assert one.labels[0] == one.labels[1]
    assert one.labels[3] == one.labels[4]
    assert one.labels[0] != one.labels[3]


def test_similarity():
    a = {1: 1, 2: 2, 3: 3}
    same = {1: 1, 2: 2, 3: 3}
    inverse = {1: 3, 2: 2, 3: 1}
    assert pearson_similarity(a, same) == pytest.approx(1)
    assert pearson_similarity(a, inverse) == pytest.approx(-1)
    assert jaccard_similarity(a, inverse) == pytest.approx(1)
    assert combined_similarity(a, inverse, .3) == pytest.approx(.4)
    assert pearson_similarity({1: 5}, {1: 5}) == 0
    assert jaccard_similarity({}, {}) == 0
    with pytest.raises(ValueError):
        combined_similarity(a, same, 1.1)


@pytest.mark.parametrize("clustered,kind", [
    (True, "combined"), (False, "pearson"), (True, "pearson")
])
def test_end_to_end(clustered, kind):
    model = KURCF(n_clusters=2, seed=7, n_neighbors=3,
                  clustered=clustered, similarity=kind).fit(TRAIN, MOVIES)
    assert model.predict(1, 1) == 5
    assert 1 <= model.predict(1, 3) <= 5
    assert model.predict(999, 1) == pytest.approx(4.5)
    recommended = model.recommend(1, 3)
    assert len(recommended) == 3
    assert not ({1, 2} & {item for item, _ in recommended})
    assert all(other != 1 for other, _ in model.neighbors(1))
    scores = rating_metrics(model, TEST)
    ranking = ranking_metrics(model, TEST, top_k=2, max_users=None)
    assert math.isfinite(scores["mae"]) and scores["n"] == len(TEST)
    assert 0 <= ranking["precision_at_k"] <= 1
    assert 0 <= ranking["recall_at_k"] <= 1


def test_split():
    rows = [(user, movie, float(movie % 5 + 1))
            for user in range(1, 6) for movie in range(1, 11)]
    a, b = user_stratified_split(rows, seed=17)
    assert (a, b) == user_stratified_split(rows, seed=17)
    assert set(a).isdisjoint(set(b))
    assert set(a) | set(b) == set(rows)
    assert {row[0] for row in a} == set(range(1, 6))


def test_movie_file_loader(tmp_path):
    folder = tmp_path / "ml-100k"
    folder.mkdir()
    movie1 = "1|Film A|01-Jan-1995||url|" + "|".join(["1"] + ["0"] * 18) + "\n"
    movie2 = "2|Film B|01-Jan-1995||url|" + "|".join(["0", "1"] + ["0"] * 17) + "\n"
    (folder / "u.item").write_text(movie1 + movie2, encoding="latin-1")
    (folder / "u.data").write_text("1\t1\t5\t9\n2\t2\t4\t10\n", encoding="ascii")
    rows, genres = load_movielens("ml-100k", tmp_path)
    assert len(rows) == 2 and len(genres[1]) == 19 and genres[1][0] == 1


def test_safe_download(tmp_path, monkeypatch):
    from pathlib import Path
    from zipfile import ZipFile
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("ml-100k/u.data", "1\t1\t4\t0\n")
        zipped.writestr("ml-100k/u.item", "placeholder\n")
    def mock_urlretrieve(_url, destination):
        Path(destination).write_bytes(archive.read_bytes())
    monkeypatch.setattr("kurcf.data.urllib.request.urlretrieve", mock_urlretrieve)
    dest = download_movielens("ml-100k", tmp_path / "datasets")
    assert (dest / "u.data").exists() and (dest / "u.item").exists()


def test_cli():
    args = build_parser().parse_args(["benchmark", "--dataset", "ml-100k", "--top-k", "5"])
    assert args.top_k == 5 and args.alpha == .3
