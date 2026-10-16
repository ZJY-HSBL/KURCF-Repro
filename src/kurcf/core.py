"""KUR-CF: variance-weighted user-attribute clustering and co-rating CF.

Independent reproduction of Zhang et al., Scientific Reports 15, 19600 (2025).
Unspecified paper details (initialization, tie breaking, prediction, fallbacks)
are explicit engineering decisions, not claims about the authors' original code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np


def variance_weights(features: np.ndarray) -> np.ndarray:
    """Equations (4)-(5): nonnegative population variance normalized to sum 1."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    variance = np.var(x, axis=0, ddof=0)
    total = float(variance.sum())
    return variance / total if total > 0 else np.full(x.shape[1], 1.0 / x.shape[1])


def weighted_squared_distances(x: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Squared equation (6); avoids a square root without affecting assignments."""
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2 * weights, axis=2)


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    inertia: float
    iterations: int


def weighted_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 42,
    max_iter: int = 100,
    projected_centers: bool = True,
) -> ClusterResult:
    """Variance-weighted K-means with deterministic farthest-first initialization.

    By default cluster means are projected onto an observed sample, matching
    the paper's closest-sample center wording (page 6). Initialization is an
    engineering choice because the paper does not specify a selection rule.
    """
    x = np.asarray(features, dtype=np.float64)
    weights = variance_weights(x)
    n, _ = x.shape
    if not isinstance(n_clusters, int) or not 1 <= n_clusters <= n:
        raise ValueError("n_clusters must be between 1 and number of samples")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    best = np.sum((x - x[first]) ** 2 * weights, axis=1)
    while len(chosen) < n_clusters:
        best[chosen] = -1.0
        next_idx = int(np.argmax(best))
        chosen.append(next_idx)
        best = np.minimum(best, np.sum((x - x[next_idx]) ** 2 * weights, axis=1))
    centers = x[np.asarray(chosen)].copy()
    previous = None
    for iteration in range(1, max_iter + 1):
        distances = weighted_squared_distances(x, centers, weights)
        labels = np.argmin(distances, axis=1)
        # Repair empty clusters by moving an existing sample furthest from
        # its center; never remove the only sample from a cluster.
        counts = np.bincount(labels, minlength=n_clusters)
        for empty in np.flatnonzero(counts == 0):
            assignment_error = distances[np.arange(n), labels].copy()
            assignment_error[counts[labels] <= 1] = -1.0
            victim = int(np.argmax(assignment_error))
            counts[labels[victim]] -= 1
            labels[victim] = empty
            counts[empty] += 1
        updated = np.empty_like(centers)
        for c in range(n_clusters):
            members = x[labels == c]
            center = members.mean(axis=0)
            if projected_centers:
                d = np.sum((members - center) ** 2 * weights, axis=1)
                center = members[int(np.argmin(d))]
            updated[c] = center
        if previous is not None and np.array_equal(labels, previous) and np.allclose(
            centers, updated, atol=1e-12, rtol=0
        ):
            centers = updated
            break
        previous = labels.copy()
        centers = updated
    # Labels are recomputed only when no empty cluster would be produced; the
    # labels returned describe the final training partition.
    inertia = float(np.sum(np.sum((x - centers[labels]) ** 2 * weights, axis=1)))
    return ClusterResult(labels.copy(), centers.copy(), weights.copy(), inertia, iteration)


def jaccard_similarity(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    """Equation (9): overlap in rated-item identities, not agreement in scores."""
    a, b = set(left), set(right)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def pearson_similarity(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    """Pearson correlation over co-rated items; insufficient overlap => zero."""
    overlap = sorted(left.keys() & right.keys())
    if len(overlap) < 2:
        return 0.0
    a = np.fromiter((left[i] for i in overlap), dtype=float)
    b = np.fromiter((right[i] for i in overlap), dtype=float)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def combined_similarity(left: Mapping[int, float], right: Mapping[int, float], alpha: float = 0.3) -> float:
    """Equation (10): alpha*Pearson + (1-alpha)*Jaccard."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * pearson_similarity(left, right) + (1 - alpha) * jaccard_similarity(left, right)


class KURCF:
    """User-based CF with optional variance-weighted user clustering.

    Input interactions must be TRAINING interactions only. Movie attributes
    are public item metadata, not extracted from the held-out test ratings.
    """

    def __init__(
        self, n_clusters: int = 7, n_neighbors: int = 20, alpha: float = 0.3,
        seed: int = 42, *, clustered: bool = True,
        similarity: str = "combined", projected_centers: bool = True,
    ) -> None:
        if n_neighbors < 1 or n_clusters < 1:
            raise ValueError("n_neighbors and n_clusters must be positive")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if similarity not in ("combined", "pearson", "jaccard"):
            raise ValueError("unknown similarity type")
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.seed = seed
        self.clustered = clustered
        self.similarity = similarity
        self.projected_centers = projected_centers
        self._fitted = False

    def fit(
        self,
        ratings: Iterable[tuple[int, int, float]],
        item_features: Mapping[int, Iterable[float]],
    ) -> "KURCF":
        attrs = {int(item): np.asarray(list(vector), dtype=float) for item, vector in item_features.items()}
        if not attrs:
            raise ValueError("item_features cannot be empty")
        d = len(next(iter(attrs.values())))
        if d == 0 or any(len(x) != d or not np.isfinite(x).all() for x in attrs.values()):
            raise ValueError("item attributes must have consistent positive dimensions and finite values")
        self.ratings: dict[int, dict[int, float]] = {}
        item_ratings: dict[int, list[float]] = {}
        for user, item, score in ratings:
            user, item, score = int(user), int(item), float(score)
            if not np.isfinite(score) or score <= 0:
                raise ValueError("training ratings must be finite and positive")
            if item not in attrs:
                raise ValueError(f"movie {item} has no attributes")
            self.ratings.setdefault(user, {})[item] = score
        if not self.ratings:
            raise ValueError("training ratings cannot be empty")
        for scores in self.ratings.values():
            for item, score in scores.items():
                item_ratings.setdefault(item, []).append(score)
        self.user_ids = sorted(self.ratings)
        self.item_ids = sorted(attrs)
        self.item_features = attrs
        self.item_mean = {i: float(np.mean(values)) for i, values in item_ratings.items()}
        self.item_counts = {i: len(values) for i, values in item_ratings.items()}
        all_scores = [score for scores in self.ratings.values() for score in scores.values()]
        self.global_mean = float(np.mean(all_scores))
        self.min_rating, self.max_rating = min(all_scores), max(all_scores)
        self.user_mean = {u: float(np.mean(list(scores.values()))) for u, scores in self.ratings.items()}
        self.clusters: ClusterResult | None = None
        if self.clustered:
            # Equation (3): Z = X @ Y; X is sparse ratings and Y is item genres.
            profiles = np.empty((len(self.user_ids), d), dtype=np.float64)
            for idx, user in enumerate(self.user_ids):
                vec = np.zeros(d, dtype=np.float64)
                for item, score in self.ratings[user].items():
                    vec += score * attrs[item]
                profiles[idx] = vec
            self.user_profiles = profiles
            self.clusters = weighted_kmeans(
                profiles, min(self.n_clusters, len(self.user_ids)),
                seed=self.seed, projected_centers=self.projected_centers
            )
            self.group = {u: int(label) for u, label in zip(self.user_ids, self.clusters.labels)}
            self.group_members: dict[int, list[int]] = {}
            for u, group in self.group.items():
                self.group_members.setdefault(group, []).append(u)
        else:
            self.group = {u: 0 for u in self.user_ids}
            self.group_members = {0: self.user_ids}
        self._neighbor_cache: dict[int, list[tuple[int, float]]] = {}
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before prediction")

    def neighbors(self, user: int) -> list[tuple[int, float]]:
        self._check_fitted()
        if user not in self.ratings:
            return []
        if user not in self._neighbor_cache:
            candidates = self.group_members[self.group[user]]
            current = self.ratings[user]
            scored = []
            for other in candidates:
                if other == user:
                    continue
                if self.similarity == "pearson":
                    sim = pearson_similarity(current, self.ratings[other])
                elif self.similarity == "jaccard":
                    sim = jaccard_similarity(current, self.ratings[other])
                else:
                    sim = combined_similarity(current, self.ratings[other], self.alpha)
                if sim > 0:
                    scored.append((other, float(sim)))
            scored.sort(key=lambda entry: (-entry[1], entry[0]))
            self._neighbor_cache[user] = scored[:self.n_neighbors]
        return self._neighbor_cache[user]

    def predict(self, user: int, item: int) -> float:
        """Mean-centered weighted neighbor rating, with item/global fallback.

        Rating prediction was not specified mathematically in the paper:
        mean-centering and fallbacks are documented reproduction choices.
        """
        self._check_fitted()
        if user in self.ratings and item in self.ratings[user]:
            return self.ratings[user][item]
        fallback = self.item_mean.get(item, self.global_mean)
        if user not in self.ratings:
            return fallback
        weighted_sum = 0.0
        total_weight = 0.0
        for neighbor, sim in self.neighbors(user):
            score = self.ratings[neighbor].get(item)
            if score is not None:
                weighted_sum += sim * (score - self.user_mean[neighbor])
                total_weight += abs(sim)
        if total_weight <= 0:
            return fallback
        prediction = self.user_mean[user] + weighted_sum / total_weight
        return float(np.clip(prediction, self.min_rating, self.max_rating))

    def recommend(self, user: int, top_k: int = 10) -> list[tuple[int, float]]:
        self._check_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        seen = set(self.ratings.get(user, {}))
        # Exact ranking among all catalog items. This is slower but avoids
        # neighbor-only candidate selection bias in precision/recall evaluation.
        candidates = (i for i in self.item_ids if i not in seen)
        ranked = [(item, self.predict(user, item)) for item in candidates]
        ranked.sort(key=lambda entry: (-entry[1], -self.item_counts.get(entry[0], 0), entry[0]))
        return ranked[:top_k]
