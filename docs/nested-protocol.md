# Nested validation and full eligible-user ranking (v0.3.0)

This is an independent reproduction protocol, not an assertion that the
original Scientific Reports paper used these exact partitions or predictions.

## Why this protocol exists

The v0.2.1 alpha/K sweeps were **descriptive measurements on the test set**.
They must not be used to select hyperparameters and then claim unbiased test
performance. In v0.3.0, model selection uses an **inner validation split**;
the outer test is untouched until the selection has been finalized.

## Partitions and evaluation

1. Use official MovieLens 100K ratings and public item-genre metadata.
2. For each seed 0–4, create an **outer user-stratified 80/20** train/test
   split. Every user retains at least one training rating.
3. Independently split outer training interactions **80/20** into inner
   training/validation with seed outer_seed + 100000. This corresponds to
   approximately 64% inner training, 16% validation and 20% outer test.
4. For **12 prespecified combinations** of K={4,7,10} and
   alpha={0,0.3,0.6,1}, fit on inner training and calculate **full
   validation MAE/RMSE**. Select by lowest MAE, then RMSE, smaller K,
   smaller alpha. **Outer-test scores are not used for selection.**
5. Refit the selected model from scratch on all outer training ratings.
   Compare against four prespecified models fitted to that same training
   set: fixed KUR-CF (K=7, alpha=.3), clustered Jaccard (K=7),
   unclustered Pearson–Jaccard (alpha=.3), and Pearson UserCF.
6. Evaluate **all outer-test ratings** for MAE/RMSE and **all eligible
   held-out-positive users** for Precision@10/Recall@10 (rating >=4).
   Ranking considers all catalog items unseen in outer training.
7. Archive five seed JSON files, SHA-256 split fingerprints, full validation
   candidate results, selected parameters, outer-test metrics, paired
   differences and exploratory 95% bootstrap CIs (4000 seed resamples).

Only **rating MAE** drives selection; this does not optimize Top-K quality.
Five random splits of one dataset are **not** five independent datasets.

## Reproduction

~~~bash
python -m pip install -e '.[dev]'
kurcf download --dataset ml-100k
kurcf nested --dataset ml-100k --seed 0 --test-ratio 0.2 \
  --validation-ratio 0.2 --fixed-clusters 7 --fixed-alpha 0.3 \
  --neighbors 20 --alpha-grid 0 0.3 0.6 1.0 \
  --cluster-grid 4 7 10 --ranking-users 0 --top-k 10 \
  --output results/seed-0.json
~~~

The value --ranking-users 0 means **all eligible users**. A positive
value samples users for local smoke tests and must not be mixed with
full-ranking results.

[Automated five-seed workflow](../.github/workflows/movielens-nested.yml)
commits raw observations and deterministic reports only after all jobs
complete. Expected artifacts are:

- benchmarks/nested/runs/seed-0.json through seed-4.json
- benchmarks/nested/ml100k-summary.json
- benchmarks/nested/README.md
- assets/figures/ml100k-nested-overview.svg

The original paper did not fully specify its train/test partitions, exact
rating prediction equation or fallbacks. Numerical agreement with the
paper cannot be asserted. Do not fabricate results if jobs fail.
