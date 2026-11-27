"""Step 3: model search on a shared subset, choice of the winner, and its refit on all training data.

    search   the same 204,800 random training rows for every model, scored on the validation set
             decision tree     grid: max depth 10/15/20 x min leaf 20/100
             random forest     200 trees; grid: features per split sqrt(p) / 30% x min leaf 10/20/50
             gradient boosting (scikit-learn HGB), XGBoost, LightGBM: setting 0 (config.SETTING_0) + 7 random
             draws each. Boosting stops when validation accuracy has not improved for 50 rounds (at most 3,000) and
             keeps its best round.  -> results/tables/search.csv, models/search_<model>.joblib (best of each)
    select   the highest validation accuracy wins; models within 0.3 points of it count as tied, and among tied
             models the lower validation log loss wins  -> results/tables/winner.json
    final    the winner refitted on all 4,947,968 training rows (boosting: patience 100, at most 3,000 rounds)
             -> models/final_<model>.joblib

Usage:  uv run python -m kinetic_mech.modeling.train [--steps search,select,final]
"""
import argparse
import json
import time
import warnings

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .. import config as C
from ..features import COLUMNS, clean, keep_columns
from .evaluate import load_set, score

warnings.filterwarnings('ignore', message='.*valid feature names.*')

MODELS = ['decision_tree', 'random_forest', 'hgb', 'xgboost', 'lightgbm']
NAMES = {'decision_tree': 'Decision tree', 'random_forest': 'Random forest', 'hgb': 'Gradient boosting (scikit-learn)',
         'xgboost': 'XGBoost', 'lightgbm': 'LightGBM'}

# random-search spaces of the three boosting libraries (7 draws each, seeds 101, 102, 103)
HGB_SPACE = {'learning_rate': loguniform(0.03, 0.3), 'max_leaf_nodes': randint(15, 128),
             'min_samples_leaf': [20, 50, 100, 200], 'l2_regularization': loguniform(1e-3, 10),
             'max_features': [0.5, 0.8, 1.0]}
XGB_SPACE = {'learning_rate': loguniform(0.03, 0.3), 'max_depth': randint(4, 11), 'min_child_weight': [1, 5, 10, 20],
             'subsample': [0.7, 0.85, 1.0], 'colsample_bynode': [0.5, 0.8, 1.0], 'reg_lambda': loguniform(1e-3, 10)}
LGB_SPACE = {'learning_rate': loguniform(0.03, 0.3), 'num_leaves': randint(15, 128),
             'min_child_samples': [20, 50, 100, 200], 'reg_lambda': loguniform(1e-3, 10),
             'feature_fraction_bynode': [0.5, 0.8, 1.0], 'subsample': [0.7, 0.85, 1.0]}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def plain(d):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items()}


def settings():
    """Every setting tried, per model."""
    s0 = C.SETTING_0
    xgb0 = {'learning_rate': s0['learning_rate'], 'grow_policy': 'lossguide', 'max_leaves': s0['max_leaf_nodes'],
            'max_depth': 0, 'min_child_weight': 1, 'reg_lambda': s0['l2_regularization'],
            'colsample_bynode': s0['max_features'], 'subsample': 1.0}
    lgb0 = {'learning_rate': s0['learning_rate'], 'num_leaves': s0['max_leaf_nodes'],
            'min_child_samples': s0['min_samples_leaf'], 'reg_lambda': s0['l2_regularization'],
            'feature_fraction_bynode': s0['max_features'], 'subsample': 1.0}
    draws = lambda space, seed: [plain(p) for p in ParameterSampler(space, 7, random_state=seed)]   # noqa: E731
    return {'decision_tree': list(ParameterGrid({'max_depth': [10, 15, 20], 'min_samples_leaf': [20, 100]})),
            'random_forest': list(ParameterGrid({'max_features': ['sqrt', 0.3], 'min_samples_leaf': [10, 20, 50]})),
            'hgb': [dict(s0)] + draws(HGB_SPACE, 101),
            'xgboost': [xgb0] + draws(XGB_SPACE, 102),
            'lightgbm': [lgb0] + draws(LGB_SPACE, 103)}


def build(model, p, patience, full_size=False):
    p = dict(p)
    if model == 'decision_tree':
        return DecisionTreeClassifier(random_state=C.SEED, **p)
    if model == 'random_forest':
        # in a full-size refit each tree still sees N_SEARCH rows (fully grown trees on 4.9M rows do not fit in memory)
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=C.SEED,
                                      max_samples=C.N_SEARCH if full_size else None, **p)
    if model == 'hgb':
        return HistGradientBoostingClassifier(max_iter=C.MAX_ROUNDS, early_stopping=True, n_iter_no_change=patience,
                                              scoring='accuracy', random_state=C.SEED, **p)
    if model == 'xgboost':
        return XGBClassifier(n_estimators=C.MAX_ROUNDS, tree_method='hist', objective='multi:softprob',
                             eval_metric='merror', early_stopping_rounds=patience, n_jobs=C.N_JOBS,
                             random_state=C.SEED, **p)
    if model == 'lightgbm':
        return LGBMClassifier(n_estimators=C.MAX_ROUNDS, objective='multiclass', metric='multi_error',
                              subsample_freq=1 if p.get('subsample', 1.0) < 1 else 0, n_jobs=C.N_JOBS,
                              random_state=C.SEED, verbose=-1, **p)
    raise ValueError(model)


def fit_hgb(model, X, y, Xv, yv):
    """scikit-learn gradient boosting with early stopping on the validation set, keeping only the trees up to the
    best round (as the other libraries do)."""
    model.fit(X, y, X_val=Xv, y_val=yv)
    best = max(1, int(np.argmax(model.validation_score_)))       # validation_score_[k]: after k rounds
    if best < model.n_iter_:
        model._predictors = model._predictors[:best]
        model.train_score_ = model.train_score_[:best + 1]
        model.validation_score_ = model.validation_score_[:best + 1]
    return model


def fit(model_name, model, X, y, Xv, yv, patience):
    """Fit; boosting stops on validation accuracy and keeps its best round. Returns the model and its rounds."""
    if model_name == 'hgb':
        model = fit_hgb(model, X, y, Xv, yv)
        return model, model.n_iter_
    if model_name == 'xgboost':
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        return model, model.best_iteration + 1
    if model_name == 'lightgbm':
        model.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lightgbm.early_stopping(patience, verbose=False)])
        return model, model.best_iteration_
    model.fit(X, y)
    return model, None


def train_matrix(rows=None, dtype=np.float32):
    """The 84 model columns of the training features: the given rows, or all 4,947,968."""
    F = np.load(C.PROCESSED / 'train_features.npy', mmap_mode='r')
    idx = [COLUMNS.index(c) for c in keep_columns()]
    y = np.load(C.PROCESSED / 'train_labels.npy')
    if rows is not None:
        return clean(np.asarray(F[rows])[:, idx]).astype(dtype), y[rows]
    X = np.empty((len(F), len(idx)), dtype=dtype)
    for s in range(0, len(F), 250_000):
        X[s:s + 250_000] = F[s:s + 250_000][:, idx]
    return clean(X).astype(dtype), y


def search_rows():
    """The 204,800 training rows shared by every model in the search (and by the dummy baseline)."""
    n = C.N_TRAIN_BATCHES * C.BATCH
    return np.sort(np.random.default_rng(C.SEED).choice(n, C.N_SEARCH, replace=False))


# ------------------------------------------------------------------------------------------ steps
def step_search():
    (C.MODELS).mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    out = C.TABLES / 'search.csv'
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=['model', 'setting'])
    rows = done.to_dict('records')
    X, y = train_matrix(search_rows())
    Xv, yv = load_set('validation')
    log('search: %s training rows, %d columns; validation %s rows' % (format(len(y), ','), X.shape[1],
                                                                       format(len(yv), ',')))
    for name, plist in settings().items():
        best = done[done.model == name].val_accuracy.max() if (done.model == name).any() else -1
        for k, p in enumerate(plist):
            if ((done.model == name) & (done.setting == k)).any():
                continue
            t0 = time.time()
            model, rounds = fit(name, build(name, p, C.PATIENCE_SEARCH), X, y, Xv, yv, C.PATIENCE_SEARCH)
            s = score(model, Xv, yv)[0]
            rows.append({'model': name, 'setting': k, 'params': json.dumps(p), 'val_accuracy': s['accuracy'],
                         'val_logloss_bits': s['logloss_bits'], 'rounds': rounds, 'fit_min': (time.time() - t0) / 60})
            pd.DataFrame(rows).to_csv(out, index=False)
            if s['accuracy'] > best:
                best = s['accuracy']
                joblib.dump({'model': model, 'params': p, 'setting': k}, C.MODELS / ('search_%s.joblib' % name))
            log('  %-14s setting %d: validation accuracy %.4f, log loss %.3f bits, rounds %s'
                % (name, k, s['accuracy'], s['logloss_bits'], rounds))


def step_select():
    """Highest validation accuracy; models within 0.3 points of it are tied and the lower log loss wins."""
    r = pd.read_csv(C.TABLES / 'search.csv')
    best = (r.sort_values(['val_accuracy', 'val_logloss_bits'], ascending=[False, True])
             .groupby('model', sort=False).head(1).set_index('model'))
    top = best.val_accuracy.max()
    tied = best[best.val_accuracy >= top - C.TIE_POINTS / 100]
    win = tied.val_logloss_bits.idxmin()
    w = {'model': win, 'setting': int(best.loc[win, 'setting']), 'params': json.loads(best.loc[win, 'params']),
         'val_accuracy': float(best.loc[win, 'val_accuracy']),
         'val_logloss_bits': float(best.loc[win, 'val_logloss_bits']),
         'rule': 'highest validation accuracy; within %.1f points counts as tied; lower validation log loss wins'
                 % C.TIE_POINTS,
         'tied_models': list(tied.index),
         'model_best': best[['setting', 'val_accuracy', 'val_logloss_bits', 'rounds']].reset_index().to_dict('records')}
    with open(C.TABLES / 'winner.json', 'w') as f:
        json.dump(w, f, indent=1)
    log('select: %s setting %d (validation %.4f, %.3f bits); tied within %.1f points: %s'
        % (win, w['setting'], w['val_accuracy'], w['val_logloss_bits'], C.TIE_POINTS, w['tied_models']))
    return w


def step_final():
    w = json.load(open(C.TABLES / 'winner.json'))
    name = w['model']
    X, y = train_matrix(dtype=np.float64 if name == 'hgb' else np.float32)
    Xv, yv = load_set('validation')
    log('final: %s on %s rows, %d columns' % (name, format(len(y), ','), X.shape[1]))
    t0 = time.time()
    model, rounds = fit(name, build(name, w['params'], C.PATIENCE_FINAL, full_size=True), X, y, Xv, yv,
                        C.PATIENCE_FINAL)
    joblib.dump({'model': model, 'params': w['params'], 'rounds': rounds, 'n_train': len(y)},
                C.MODELS / ('final_%s.joblib' % name))
    log('  fitted in %.1f min, rounds kept %s' % ((time.time() - t0) / 60, rounds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', default='search,select,final')
    args = ap.parse_args()
    for s in args.steps.split(','):
        {'search': step_search, 'select': step_select, 'final': step_final}[s]()


if __name__ == '__main__':
    main()
