# Models

Fitted models are written here by `python -m kinetic_mech.modeling.train` and are not in the repository (the final
LightGBM model is about 340 MB):

| File | Model |
|---|---|
| `search_<model>.joblib` | best setting of each model in the search (204,800 training rows) |
| `final_lightgbm.joblib` | the winner, refitted on all 4,947,968 training rows (1,940 trees) |

To use models saved elsewhere: `$env:KMC_MODELS_DIR = "D:\path\to\models"`.
