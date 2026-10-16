# Reproduction audit / 方法审计

Source: Zhang et al., Scientific Reports 15:19600 (2025), DOI:10.1038/s41598-025-96705-0.

| Pages / equations | Paper statement | Implementation choice / limitation |
|---|---|---|
| p5, Eq.3 | Z=X@Y user attribute scores | Raw train ratings multiplied by item genre binary attributes; unnormalized rating activity matters |
| p5, Eqs.4–5 | variance weights | Population variance normalized to sum one; Eq.4 denominator notation is ambiguous |
| p5, Eq.6 | weighted Euclidean clustering distance | Weighted squared distance for argmin; mathematically equivalent |
| p6, Eq.7 | cluster mean point; choose closest actual sample if needed | projected cluster mean; no exact initialization method given |
| p6, Eq.8 | silhouette / contour | standard sampled silhouette in weighted space (not claimed identical) |
| p6, Eqs.9–10 | co-rated item overlap and alpha Pearson+(1-alpha)Jaccard | exact stated formula with alpha=0.3; Eq.10 does not explicitly contain K-means distance |
| p7, prediction | score unobserved items from similar neighbors | mean-centered user CF with clipping and item/global fallback; equation not supplied in paper |
| p8–9, cluster tuning | 100k k=7, 1M k=6 | use as default values, no claim of independent verification |
| p14–19, plots | comparative errors, P/R, running time | benchmarking is newly implemented with a disclosed protocol; numerical paper claims unverified |

## Explicit non-paper conventions

1. Deterministic farthest-first center initialization with seed 42 because the paper does not specify a complete initialization procedure.
2. Seeded user-wise randomized 80/20 split; holdout ratings never enter user profiles, means or neighbor similarities.
3. Pearson computed on common rated items; fewer than two overlaps or zero variance returns zero.
4. Top-N positive-similarity neighbors in the same cluster (no negative Pearson neighbors).
5. Mean-centered weighted neighborhood prediction; fallback to item mean or global mean on missing support.
6. Default 3000 held-out rating samples and 100 ranking users, to bound expensive evaluation; full numbers appear in output.
7. Relevant items have held-out rating at least 4. Precision@K / Recall@K are micro-aggregated over users with at least one positive test item.
8. Recommendations score all catalog items not rated in training, not a prefiltered neighbor candidate set.

## Next research steps

Match author-provided train/test splits (if obtainable), repeat multi-seed trials with confidence intervals, add ItemCF and SVD, test genuine cold-start cohorts, quantify candidate-selection effects and add unit tests based on externally verifiable examples. Do not claim 60% or 65% improvements without a fair, independently rerun benchmark.
