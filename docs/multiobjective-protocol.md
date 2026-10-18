# v0.4.0 — Multi-objective validation selection (initial implementation)

The verified v0.3.1 MovieLens 100K experiment showed that validation-MAE
selection (K=4, alpha=0 in all five splits) reduced outer-test MAE but also
reduced Recall@10. The unclustered hybrid achieved the highest observed
Recall@10 among the tested methods. These are observations, **not**
permission to tune hyperparameters on the outer test.

This first v0.4.0 step adds a deterministic **validation-only**
multi-objective selection component in `kurcf.objectives`:

- `select_objectives(rows)["rating_mae"]`: minimum validation MAE,
  with RMSE and smaller parameter tie breaks.
- `select_objectives(rows)["ranking_recall"]`: maximum validation
  Recall@K, with Precision@K, MAE and parameter tie breaks.
- `pareto_frontier(rows)`: nondominated candidates minimizing
  validation MAE and maximizing validation Recall@K.

A candidate row must contain K, alpha, validation MAE, RMSE,
Precision@K and Recall@K. The selector rejects missing and nonfinite
metrics. This commit does **not** yet produce a new measured MovieLens
benchmark: a complete experiment must first compute ranking metrics
on the **inner validation split**, refit both chosen configurations
on the full outer training split, and compare their performance on
an untouched outer test. The five previously reported outer-test
results must not be reused as tuning data.

Research plan: extend the predeclared grid to K=2,3,4,7,10 and
alpha=0,0.05,0.1,0.3,0.6,1; use the same inner-validation
candidate ranking population and record its size, evaluation runtime
and split fingerprints. Only after these controls are implemented
should five-seed results and SVG charts be published.

## Implemented experimental pipeline

The `kurcf multiobjective` CLI now runs the two inner-validation
selection objectives, refits both models, and evaluates five matched
outer-test baselines. The automated GitHub Actions workflow
`movielens-multiobjective.yml` uses five seeds (0–4), a prespecified
30-point grid, **120 sampled eligible inner-validation users per
candidate**, and **all eligible outer-test users**. It archives
original JSON, parameter choices, bootstrap intervals and an SVG
chart after all five jobs finish. Sampling the validation population
reduces compute, but introduces ranking-selection variance; this
limitation must be reported alongside results. No v0.4.0 metrics are
claimed until the workflow actually succeeds.
