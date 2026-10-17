# MovieLens 100K — full rating evaluation and ablation study

**Actual measured results** from [raw per-seed JSON](runs/), not paper figures.

Five independent stratified splits: seeds [0, 1, 2, 3, 4].
Rating evaluation covers the **entire held-out set** per seed: [20000, 20000, 20000, 20000, 20000].
Top-10 ranking covers **sampled users** per seed: [100, 100, 100, 100, 100].
Settings: K=7, N=20, α=0.3, relevant rating ≥ 4.
95% bootstrap intervals are resampled over seeds (4,000 draws); five seeds are exploratory.

![Baseline and ablation results](../../assets/figures/ml100k-study-overview.svg)

## Mean ± sample standard deviation across seeds

| Variant | MAE ↓ | RMSE ↓ | Precision@10 (%) ↑ | Recall@10 (%) ↑ |
|:--|--:|--:|--:|--:|
| KUR-CF | 0.8139 ± 0.0041 | 1.0457 ± 0.0057 | 2.2000 ± 0.4062 | 1.9264 ± 0.3112 |
| UserCF | 0.8633 ± 0.0047 | 1.0954 ± 0.0047 | 2.4800 ± 0.5070 | 2.1941 ± 0.4848 |
| Cluster-Pearson | 0.8336 ± 0.0021 | 1.0683 ± 0.0019 | 2.3600 ± 0.8620 | 2.0497 ± 0.6747 |
| Cluster-Jaccard | 0.8095 ± 0.0048 | 1.0374 ± 0.0047 | 1.9800 ± 0.3421 | 1.7402 ± 0.2895 |
| Unclustered hybrid | 0.8339 ± 0.0032 | 1.0693 ± 0.0048 | 3.2400 ± 1.2095 | 2.7921 ± 0.8661 |
| Uniform weights | 0.8157 ± 0.0048 | 1.0473 ± 0.0074 | 2.1400 ± 0.4393 | 1.8772 ± 0.3455 |
| Mean centers | 0.8131 ± 0.0021 | 1.0446 ± 0.0032 | 2.2600 ± 0.5273 | 1.9760 ± 0.3832 |

[Full JSON summary](ml100k-summary.json) contains seed-level split fingerprints, 95% CIs, runtime and paired baseline-minus-KUR-CF differences.

## Sensitivity sweeps

![Alpha sensitivity](../../assets/figures/ml100k-study-alpha.svg)

![Cluster-count sensitivity](../../assets/figures/ml100k-study-clusters.svg)

The alpha and K sweeps use the held-out test set only for **descriptive analysis**. Do not select a best setting using these plots and report its test metric as unbiased; that requires a separate validation split.

## Interpretation limits

This is an independent protocol, not a strict numerical replication of the paper. The original data partitions and prediction details are not public in the paper. The full held-out **rating** set is evaluated, while Top-K ranking uses 100 sampled eligible users per seed. Paired multi-seed differences and CIs do not prove the paper's reported improvements.

To regenerate figures and summary from committed raw observations:

~~~bash
python scripts/summarize_study.py --min-seeds 5
python scripts/summarize_study.py --min-seeds 5 --check
~~~
