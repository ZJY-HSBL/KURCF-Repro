"""MovieLens 100K / 1M loaders and reproducible user-stratified splits."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import urllib.request
import zipfile

Rating = tuple[int, int, float]

DATASETS = {
    "ml-100k": ("https://files.grouplens.org/datasets/movielens/ml-100k.zip", ("u.data", "u.item")),
    "ml-1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", ("ratings.dat", "movies.dat")),
}


def download_movielens(name: str, root: str | Path = "data") -> Path:
    """Download from GroupLens; extract only required files (no data in git)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name!r}")
    root = Path(root)
    location = root / name
    needed = DATASETS[name][1]
    if all((location / file).exists() for file in needed):
        return location
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{name}.zip"
    location.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urllib.request.urlretrieve(DATASETS[name][0], archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for filename in needed:
            member = f"{name}/{filename}"
            # Whitelist known paths; do not extract untrusted paths.
            with archive.open(member) as source, (location / filename).open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
    return location


def load_movielens(name: str, root: str | Path = "data") -> tuple[list[Rating], dict[int, list[float]]]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name!r}")
    location = Path(root) / name
    if not all((location / filename).exists() for filename in DATASETS[name][1]):
        raise FileNotFoundError(f"{location} is missing; run 'kurcf download --dataset {name}'")
    ratings: list[Rating] = []
    attrs: dict[int, list[float]] = {}
    if name == "ml-100k":
        with (location / "u.item").open(encoding="latin-1") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 24:
                    raise ValueError(f"malformed u.item row: {line[:60]!r}")
                attrs[int(fields[0])] = [float(value) for value in fields[5:24]]
        with (location / "u.data").open(encoding="ascii") as stream:
            for line in stream:
                user, item, score, *_ = line.split()
                ratings.append((int(user), int(item), float(score)))
    else:
        movie_genres: dict[int, set[str]] = {}
        with (location / "movies.dat").open(encoding="latin-1") as stream:
            for line in stream:
                movie_id, _title, genre_text = line.rstrip("\n").split("::", 2)
                movie_genres[int(movie_id)] = set(genre_text.split("|"))
        genres = sorted(set().union(*movie_genres.values()))
        index = {g: idx for idx, g in enumerate(genres)}
        for movie, assigned in movie_genres.items():
            features = [0.0] * len(genres)
            for genre in assigned:
                features[index[genre]] = 1.0
            attrs[movie] = features
        with (location / "ratings.dat").open(encoding="ascii") as stream:
            for line in stream:
                user, item, score, _timestamp = line.strip().split("::")
                ratings.append((int(user), int(item), float(score)))
    if not ratings:
        raise ValueError(f"empty dataset: {location}")
    return ratings, attrs


def user_stratified_split(
    ratings: list[Rating], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[Rating], list[Rating]]:
    """Hold out interactions per user; keep at least one train item per user.

    Random split is an explicit reproduction convention: the paper does not
    specify the exact split, fold memberships, or random seed.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0,1)")
    grouped: dict[int, list[Rating]] = defaultdict(list)
    for row in ratings:
        grouped[int(row[0])].append(row)
    rng = random.Random(seed)
    train: list[Rating] = []
    test: list[Rating] = []
    for user in sorted(grouped):
        rows = grouped[user][:]
        rng.shuffle(rows)
        if len(rows) == 1:
            train.extend(rows)
            continue
        count = min(len(rows) - 1, max(1, round(len(rows) * test_ratio)))
        test.extend(rows[:count])
        train.extend(rows[count:])
    return train, test
