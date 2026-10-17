# MovieLens 100K — nested validation and full eligible-user ranking

**Actual measurements from [raw seed JSON](runs/).** Hyperparameters are selected on an **inner validation split**; the selected model is then refit on outer training and evaluated on the held-out outer test.

- Outer 80/20 seeds: [0, 1, 2, 3, 4].
- Inner validation fraction of outer training: 20%.
- Outer test ratings per seed: [20000, 20000, 20000, 20000, 20000] (all evaluated).
- Eligible ranking users per seed: [935, 932, 927, 929, 934] (all with held-out rating >=4, not sampled).
- All baselines use the **same outer train/test split** and ranking population.
- Selection: lowest inner-validation MAE; tie-break RMSE, K, alpha.
- 95% bootstrap intervals resample five split seeds, not individual ratings.

![Nested validation and outer-test results](../../assets/figures/ml100k-nested-overview.svg)

## Outer-test mean ± sample SD across five seeds

| Model | MAE ↓ | RMSE ↓ | Precision@10 (%) ↑ | Recall@10 (%) ↑ |
|:--|--:|--:|--:|--:|
| KUR-CF (fixed) | 0.8139 ± 0.0041 | 1.0457 ± 0.0057 | 2.6348 ± 0.1688 | 2.2142 ± 0.1448 |
| KUR-CF (validation-selected) | 0.8081 ± 0.0068 | 1.0361 ± 0.0067 | 2.1389 ± 0.1418 | 1.7974 ± 0.1190 |
| Cluster-Jaccard | 0.8095 ± 0.0048 | 1.0374 ± 0.0047 | 2.1971 ± 0.2710 | 1.8464 ± 0.2297 |
| Unclustered hybrid | 0.8339 ± 0.0032 | 1.0693 ± 0.0048 | 3.6226 ± 0.1373 | 3.0440 ± 0.1106 |
| UserCF (Pearson) | 0.8634 ± 0.0047 | 1.0955 ± 0.0047 | 2.3899 ± 0.1873 | 2.0079 ± 0.1523 |

## Inner-validation selections (not test-selected)

| Selected K | Selected alpha | Seeds |
|--:|--:|--:|
| 4 | 0.00 | 5 |

## Paired outer-test differences

**Baseline minus validation-selected KUR-CF**, paired by outer split. Positive MAE/RMSE differences favor selected KUR-CF; positive Precision/Recall differences favor the baseline.

| Baseline | ΔMAE | ΔRMSE | ΔPrecision@10 (pp) | ΔRecall@10 (pp) |
|:--|--:|--:|--:|--:|
| KUR-CF (fixed) | +0.0059 | +0.0096 | +0.4959 | +0.4168 |
| Cluster-Jaccard | +0.0014 | +0.0013 | +0.0581 | +0.0491 |
| Unclustered hybrid | +0.0258 | +0.0332 | +1.4837 | +1.2466 |
| UserCF (Pearson) | +0.0553 | +0.0594 | +0.2509 | +0.2106 |

[Machine-readable summary](ml100k-summary.json) includes exploratory 95% intervals, paired differences and SHA-256 partition fingerprints.

## Limitations

- Hyperparameter selection optimizes **rating MAE**, not ranking quality.
- Inner and outer partitions share users, but not interactions. No test ratings enter fitting or selection.
- Five partitions of **one dataset** are not five independent datasets.
- Original paper partitions and full prediction equation are unavailable.

## Rebuild

~~~bash
python scripts/summarize_nested.py --min-seeds 5
python scripts/summarize_nested.py --min-seeds 5 --check
~~~
