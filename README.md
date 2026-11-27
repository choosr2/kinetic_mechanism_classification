# Kinetic mechanism classification with engineered features

Code for the CHE1147 project paper. We classify 20 catalytic reaction mechanisms (M1–M20) from simulated
concentration–time data (Burés & Larrosa, *Nature* 2023) with LightGBM on 84 engineered kinetic features, and use
grouped permutation importance to show which kinetic tests the model relies on for each mechanism.

## Install

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.13; exact versions in `uv.lock`).

```powershell
uv sync
```

`requirements.txt` lists the same packages for pip users (`pip install -r requirements.txt`, then `pip install -e .`).

## Data

The 10 GB dataset is not in the repository. See [data/README.md](data/README.md) for where to download it and where to
put it.

## Run the analysis

Each step is a module of the `kinetic_mech` package (`src/kinetic_mech/`). Run them in this order from the repository
folder; each one reads what the step before wrote.

| Step | Command | Paper (Methods) | Writes |
|---|---|---|---|
| 1. Features | `uv run python -m kinetic_mech.make_features` | 136 features; the authors' sampling of training (2,416 batches of 2,048) and validation (batches of 25) | `data/processed/` |
| 2. EDA | `uv run python -m kinetic_mech.eda` (`--steps quality` needs only the raw data) | dataset structure (Appendix 1), data quality, class balance, missing values, Pearson correlation, correlation filter (136 → 84), feature distributions, feature medians per mechanism on the test set | `results/tables/eda/`, `results/tables/drop_correlated.json`, `results/report/Appendix_1_dataset_structure.csv` |
| 3. Models | `uv run python -m kinetic_mech.modeling.train` | dummy, decision tree, random forest, gradient boosting, XGBoost, LightGBM on 204,800 rows; tie rule; refit on 4,947,968 rows | `results/tables/search.csv`, `winner.json`, `models/` |
| 4. Evaluation | `uv run python -m kinetic_mech.modeling.evaluate` | accuracy and log loss on validation and the five released test sets (6 points with 0, 1, 5% noise; 20 points, 1%; 2 points, 1%), recall and top-2 recall per mechanism, confusion matrices | `results/tables/results.csv`, `per_class_recall.csv`, `top2_recall.csv`, `confusion_*.csv` |
| 5. Permutation | `uv run python -m kinetic_mech.permutation --noisy` | grouped permutation of the 12 families and 3 blocks, 3 repeats; recall and confusion changes; generalist/specialist roles; the noise family also on the 1% and 5% noise test sets | `results/tables/permutation/`, `permutation_noisy/` |
| 6. Figures | `uv run python -m kinetic_mech.plots` (`report` or `supporting` for one group) and `uv run python -m kinetic_mech.plots_appendix` | report Figures 1 and 2, Appendices 3 and 4; supporting figures and Figures S1–S12 | `results/report/`, `results/supporting/` |
| 7. Appendices | `uv run python -m kinetic_mech.appendix` | report Appendices 5 and 6; the supporting document | `results/report/`, `results/supporting/` |

Run times on a 16-thread desktop: step 1 about 10 minutes (2.8 GB written; 6 minutes for the features themselves), the model search a few hours, the final
LightGBM refit about 4.5 hours, the permutation step about 75 minutes. Steps 6 and 7 only need `results/` (plus the
raw data for Figure 1, Appendix 1 and Figures S1–S12).

`results/` already holds the outputs of the run reported in the paper, so the figures, the appendices and the notebook
work without re-running steps 1–5. `notebooks/01_results_walkthrough.ipynb` walks through those results.

Tests (the feature code reproduces the paper's features exactly): `uv run pytest`.

## Settings

Every number quoted in the Methods (seeds, sample sizes, search spaces, early stopping, the 0.3-point tie rule,
permutation repeats and significance thresholds) is set once in [`src/kinetic_mech/config.py`](src/kinetic_mech/config.py).

## Report → code

Everything in the report (`CHE1147_Report.docx`) and where it comes from. Files in `results/report/` carry the
report's own labels.

| Report item | Output | Code |
|---|---|---|
| Figure 1 (feature families) | `results/report/Figure_1_feature_families` | `plots.figure1_feature_families` |
| Figure 2, panels A–E (captioned "Figure 3" in the draft) | `results/report/Figure_2_models_and_permutation` | `plots.figure2_models_and_permutation` |
| Appendix 1 (dataset structure) | `results/report/Appendix_1_dataset_structure.csv` | `eda.dataset_structure` |
| Appendix 2 (the 20 mechanisms) | not generated: Burés & Larrosa (2023), Fig. 1 | – |
| Appendix 3 (feature distributions) | `results/report/Appendix_3_feature_distributions` | `plots.appendix3_feature_distributions` |
| Appendix 4 (Pearson correlation) | `results/report/Appendix_4_pearson_correlation` | `plots.appendix4_pearson_correlation` |
| Appendix 5, Tables 5A–5E (hyperparameters) | `results/report/Appendix_5_hyperparameter_optimisation.docx` | `appendix.build_b`, from `results/tables/search.csv` |
| Appendix 6, Tables 6A–6L (all 136 features) | `results/report/Appendix_6_engineered_features.docx`, `.csv` | `appendix.build_6` |

Numbers quoted in the text:

| Report text | Table (in `results/tables/`) | Code |
|---|---|---|
| 5.1 million samples; splits of 4.95 M / 50,000 / 100,000; every mechanism 5% | `eda/class_balance.csv` | `eda.class_balance` |
| loadings 1–10 mol%; no missing values or duplicates found | `eda/data_quality.csv` | `eda.data_quality` |
| 136 features, 12 families; Pearson filter → 84 features | `drop_correlated.json`, `eda/pearson_136.csv` | `features.py`, `eda.correlation_filter` |
| 3.5% of values more than 3 IQR beyond the quartiles | `eda/distribution_84.csv` (`n_extreme` / `n_values`) | `eda.distributions` |
| 2,416 blocks of 2,048 (4,947,968; 2,032 left out); 3–10, 15, 20 points; 0–2% noise; validation blocks of 25 | – | `config.py`, `dataset.batch_draws` |
| six models on 204,800 samples; grid and random search; best LightGBM setting | `search.csv`, `winner.json` | `modeling/train.py` |
| validation accuracies (5.0, 53.5, 64.4, 70.7–71.1%), log loss | `results.csv`, `search.csv` | `modeling/evaluate.py` |
| 90.5% test accuracy; noise and sampling (49.6, 85.4, 70.5%, 8.7 and 3.6 points) | `results.csv` | `modeling/evaluate.py` |
| 13 of 20 recalls above 90%; M11 65.5%, M14 48.0% | `per_class_recall.csv` | `modeling/evaluate.py` |
| top-2 recall 98.2% (M11), 97.0% (M14) | `top2_recall.csv` | `modeling/evaluate.py` |
| family accuracy losses (24.1, 20.3, 18.2, 16.6%), 15–17 mechanisms affected, 75% / 91% in deactivation | `permutation/family_summary.csv`, `runs.csv` | `permutation.py` |
| recall changes per family and mechanism (e.g. M3/M4 −49%, M1 −61%, M8 75.6 → 15.9%) | `permutation/recall_change.csv` | `permutation.py` |
| where the lost experiments go (M3/M4 → M5, M6 → M8, M8 → M2 and M6) | `permutation/confusion_shift.csv` | `permutation.py` |
| what a family measures for a mechanism (VTNA order and overlay of M8, M1, M13, M18, M19; inhibitor fit and deficit of M20 vs M10 and M14) | `eda/feature_medians_by_mechanism_test.csv` | `eda.step_mechanisms` |

## Repository layout

```
├── data/
│   ├── raw/            the authors' pickles (download; not in git)
│   └── processed/      feature matrices from step 1 (not in git)
├── models/             fitted models from step 3 (not in git)
├── notebooks/          01_results_walkthrough.ipynb
├── results/
│   ├── report/         every item of the report, named as in the report (see "Report → code")
│   ├── tables/         CSV/JSON outputs of steps 2-5 (eda/, permutation/), read by the figures and appendices
│   └── supporting/     not in the report: figures/ (Figure 2's panels on their own, square Pearson, example run),
│                       eda/ (data quality, class balance), feature_examples/ (Figures S1-S12), supporting document
├── src/kinetic_mech/
│   ├── config.py       folders, seeds, sample sizes, decision rules
│   ├── dataset.py      loading and the authors' sampler
│   ├── features.py     the 136 features, one function per family
│   ├── make_features.py
│   ├── eda.py
│   ├── modeling/       train.py (search, tie rule, refit), evaluate.py
│   ├── permutation.py
│   ├── plots.py, plots_appendix.py, style.py
│   └── appendix.py
├── tests/              test_features.py
├── pyproject.toml, uv.lock, requirements.txt
└── README.md
```

## Notes

- The feature family called "Death laws" in the code is the "Deactivation laws" family in the paper.
- Setting 0 of each boosting library is the best setting of an earlier, separate search; the other 7 per library are
  random draws (`config.SETTING_0`, `modeling/train.py`).
- The authors drew their training noise without a seed, so the exact noise values differ from theirs; the noise
  here is seeded (`config.TRAIN_NOISE_SEED`) and the results are reproducible.

## Reference

Burés J, Larrosa I (2023). Organic reaction mechanism classification using machine learning. *Nature* 613, 689–695.
