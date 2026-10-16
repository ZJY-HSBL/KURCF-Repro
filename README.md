# KURCF-Repro

Independent Python implementation of the variance-weighted K-means collaborative filtering algorithm **KUR-CF**.

**Original research:** Zhang, S., Chen, S., Yu, X. & Mei, S., *Research on collaborative filtering algorithm based on improved K-means algorithm for user attribute rating and co-rating*. **Scientific Reports** 15, 19600 (2025). https://doi.org/10.1038/s41598-025-96705-0

[中文文档](README.zh-CN.md) | [Methodology and limitations](docs/reproduction-notes.md)

> This repository is **not the original authors' source code**. Its implementation decisions are documented. The paper's reported numerical gains have **not yet been independently reproduced**.

## Quick start

Python 3.10+ required.

~~~bash
git clone https://github.com/ZJY-HSBL/KURCF-Repro.git
cd KURCF-Repro
python -m pip install -e ".[dev]"
pytest -q

kurcf download --dataset ml-100k
kurcf benchmark --dataset ml-100k --output results/ml100k.json

kurcf download --dataset ml-1m
kurcf benchmark --dataset ml-1m --output results/ml1m.json

kurcf sweep-k --dataset ml-100k --min-k 4 --max-k 12 --output results/silhouette.json
~~~

Data are downloaded from GroupLens on demand. Do not commit datasets.

## Paper-to-code mapping

| Paper | Code |
|---|---|
| Equation 3: user-attribute rating matrix Z = X @ Y | training ratings × movie-genre indicators |
| Equations 4–5: normalized attribute variances | variance_weights |
| Equation 6: variance-weighted Euclidean distance | weighted_squared_distances, weighted_kmeans |
| Equation 7: mean and nearest-sample center | weighted_kmeans(projected_centers=True) |
| Equation 9: co-rated item Jaccard overlap | jaccard_similarity |
| Equation 10: alpha Pearson + (1-alpha) Jaccard | combined_similarity (alpha = 0.3) |
| Paper-selected cluster counts | k=7 (100K), k=6 (1M) |

The model searches for nearest neighbors **within a user's assigned cluster**, then predicts ratings from those neighbors.

### Baselines

The benchmark implements: KUR-CF, unclustered Pearson UserCF, and clustered Pearson (ablation). It is not an attempt to impersonate the paper's unspecified baseline settings. ItemCF and SVD comparisons remain future extensions.

### Experimental protocol

- Fixed-seed random **per-user 80/20** train/test split; the original article does not identify exact split indices or a seed.
- All user profiles, similarities and means are fitted from **training ratings only**.
- MAE/RMSE on a documented sample of test interactions (default cap: 3000).
- Micro Precision@K / Recall@K with positive held-out ratings >= 4, over documented sampled users (default cap: 100).
- All catalog items not in a user's training history are Top-K candidates. Timings and sample sizes are included in benchmark JSON.
- Silhouette sweep uses conventional sampled silhouette on variance-scaled profiles, not an assertion that the article's center-based description is identical.

These conventions cannot be used to claim exact numerical replication of the published figures. See [reproduction notes](docs/reproduction-notes.md).

## Releasing

GitHub Actions executes pytest on Python 3.10/3.12 on pushes and pull requests. After a successful main-branch push, the release job builds a wheel and source archive, publishing the first GitHub Release for the package version (starting at v0.1.0). Already existing tags/releases are not overwritten. Subsequent releases require bumping both pyproject.toml and src/kurcf/__init__.py.

## License

Independent source code: MIT. MovieLens belongs to GroupLens and has separate usage terms; no dataset is redistributed here. For the algorithm itself cite the original paper.
