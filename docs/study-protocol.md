# Full-rating and ablation study protocol (v0.2.0)

The original 2025 Scientific Reports paper does not provide sufficient
information to reconstruct its exact train/test partitions and complete
prediction formula. This experiment is therefore an **independent controlled
evaluation**, not a claim of reproducing the article's numeric results.

## Fixed study protocol

- Dataset: official GroupLens MovieLens 100K, using the original movie genres.
- Splits: five user-stratified 80/20 partitions, seeds 0–4; no test ratings in
  user profiles, similarity, model fitting or item/global fallback statistics.
- Each seed evaluates **every held-out rating** (normally 20,000 cases).
- Top-10 recommendation is evaluated on **100 sampled eligible users** per
  seed, using all catalog items unseen in that user's training interactions.
  Relevance is a held-out rating of at least 4. This is NOT full ranking
  evaluation, and must not be described as such.
- Each variant within a seed uses the **same train/test split**, held-out
  rating sample and ranking user sample. Per-seed SHA-256 split fingerprints
  are retained.
- Main parameters: K=7, 20 neighbors, alpha=0.3, Top-10.
- Seven variants: KUR-CF; unclustered Pearson UserCF; clustered Pearson;
  clustered Jaccard; unclustered Pearson–Jaccard; uniform (rather than
  variance-derived) clustering weights; unprojected mean cluster centers.
- Exploratory alpha grid: 0, 0.3, 0.6, 1.0.
- Exploratory cluster-count grid: 4, 7, 10.
- Seed-level means are summarized with sample standard deviations and
  percentile bootstrap 95% CIs over seeds (4,000 resamples). Paired
  baseline-minus-KUR-CF differences are calculated on the same seed.

**No hyperparameter tuning on test data:** Alpha and K sensitivity curves
are descriptive. Selecting a best setting using the held-out test set and
then reporting its performance as unbiased would leak test information. A
separate validation split is required for tuning.

## Run one seed locally

~~~bash
python -m pip install -e ".[dev]"
kurcf download --dataset ml-100k
kurcf study --dataset ml-100k --seed 0 --clusters 7 \
  --neighbors 20 --alpha 0.3 --rating-cases 0 \
  --ranking-users 100 --top-k 10 --output results/seed-0.json
~~~

The default study grids are alpha = [0, 0.3, 0.6, 1] and K = [4, 7, 10].
The CLI also accepts explicit --alpha-grid and --cluster-grid values.

## Multi-seed automation

[GitHub Actions: MovieLens 100K study](../.github/workflows/movielens-study.yml)
runs five independent seeds in parallel. Each seed uploads raw JSON as a
workflow artifact. After all five succeed, the aggregation job validates
comparability, computes uncertainty intervals, and commits the raw results,
the summary JSON and three vector figures.

Generated outputs, committed after the successful five-seed run:

- benchmarks/study/runs/seed-0.json through seed-4.json
- benchmarks/study/ml100k-summary.json
- benchmarks/study/README.md
- assets/figures/ml100k-study-overview.svg
- assets/figures/ml100k-study-alpha.svg
- assets/figures/ml100k-study-clusters.svg

The five-seed study completed in [GitHub Actions run 39006215491](https://github.com/ZJY-HSBL/KURCF-Repro/actions/runs/39006215491). Raw JSON, summary and figures are committed and independently regenerable. Future failures must not be replaced by invented measurements.

## Methodological limitations

1. KUR-CF's rating prediction and fallback formula are explicit independent
   implementation conventions; the paper's original source is not available.
2. Genre-profile magnitudes depend on rating activity (raw Z = X Y sums).
3. Ranking on 100 users per seed is not a full ranking benchmark.
4. Bootstrap CIs over five split seeds quantify split sensitivity only.
5. A real hyperparameter search must use a separate validation set.
6. Running times are GitHub runner-specific, not directly comparable to the
   article's hardware.
7. MovieLens 1M currently has smoke experiments only; a full 1M protocol
   should be separately planned to manage unclustered UserCF costs.
