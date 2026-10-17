# Verified smoke runs on official MovieLens datasets

**Completed v0.3.1:** The [nested-validation and full eligible-user ranking study](nested/README.md) separates model selection from the outer test and includes original per-seed JSON, five-seed statistics and a generated SVG figure. See the [protocol](../docs/nested-protocol.md). Previous smoke and ablation measurements remain unchanged.


**Completed in v0.2.1:** The [five-seed full-rating study](study/README.md) includes seven ablations, paired bootstrap intervals, original per-seed JSON and three SVG figures. See the [documented protocol](../docs/study-protocol.md). The figures and tables below remain the original **small-sample smoke runs**, retained for provenance.

These runs were executed by GitHub Actions using the code in this repository.
They are **not** a numerical replication of the original article's plots.

The source paper is Zhang et al., Scientific Reports (2025):
https://doi.org/10.1038/s41598-025-96705-0

## Experimental setup

- Independent user-stratified seeded 80/20 split; random seed 42.
- Train/test ratings: 80,000/20,000 (100K); 800,193/200,016 (1M).
- Recommended cluster counts k=7 (100K), k=6 (1M), and alpha=0.3.
- 20 neighbors; candidate set is every training-unseen catalog movie; test relevant rating >= 4.
- Errors computed on 120 sampled test interactions for 100K, 300 for 1M.
- Top-10 ranking computed over 12 sampled users for 100K, 30 for 1M.
- All baselines use the same split and evaluation sample within each dataset.
- Timing depends on GitHub runner hardware, and is not compared to the paper's SVD timing figure.

### MovieLens 100K

| Method | MAE ↓ | RMSE ↓ | Precision@10 ↑ | Recall@10 ↑ |
|---|---:|---:|---:|---:|
| KUR-CF | 0.86246 | 1.10465 | 0.00833 | 0.01075 |
| UserCF (Pearson) | 0.89891 | 1.15737 | 0.00833 | 0.01075 |
| Cluster-Pearson | 0.89838 | 1.16141 | 0.02500 | 0.03226 |

[Raw JSON](ml100k-smoke.json) | [GitHub Actions run](https://github.com/ZJY-HSBL/KURCF-Repro/actions/runs/38940035981)

### MovieLens 1M

| Method | MAE ↓ | RMSE ↓ | Precision@10 ↑ | Recall@10 ↑ |
|---|---:|---:|---:|---:|
| KUR-CF | 0.76263 | 0.97466 | 0.01667 | 0.00916 |
| UserCF (Pearson) | 0.77581 | 0.96391 | 0.01667 | 0.00916 |
| Cluster-Pearson | 0.77987 | 0.98706 | 0.02667 | 0.01465 |

[Raw JSON](ml1m-smoke.json) | [GitHub Actions run](https://github.com/ZJY-HSBL/KURCF-Repro/actions/runs/38940286685)

## Interpretation

KUR-CF has a smaller sampled MAE on both datasets, but it **does not win every metric**: UserCF has a smaller RMSE on MovieLens 1M, while clustered Pearson has a larger sampled Top-10 precision and recall on both datasets. At 12/30 users, Top-K estimates have high sampling uncertainty and must not be presented as definitive algorithm rankings. Reported 0.1 MAE and high precision in the paper are not recovered using the disclosed independent protocol.

To assess genuine improvements, run repeated stratified splits with uncertainty intervals and fixed candidate sets, implement additional baselines, and obtain the authors' original preprocessed partitions and prediction rules. Our current design verifies source-code execution, not the paper's originally claimed effect sizes.
