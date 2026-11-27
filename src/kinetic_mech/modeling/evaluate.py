"""Step 4: score every model on the validation set and the test set.

Models: a dummy baseline that always predicts the class shares of the search rows (it ignores the features), the
best setting of each model from the search, and the final refitted winner. The winner was fixed (winner.json)
before any test score was computed.
Scores: accuracy, log loss (bits), recall of every mechanism, and for the final model its confusion matrices and its
top-2 recall (share of each mechanism's experiments whose true mechanism is among the model's two most likely).
Sets: validation and the five released test sets (6 points with 0, 1 and 5% noise; 20 points, 1%; 2 points, 1%);
--main-test-only scores the main test set (6 points, no noise) only.

Outputs: results/tables/results.csv, per_class_recall.csv, top2_recall.csv, confusion_<set>.csv
Usage:  uv run python -m kinetic_mech.modeling.evaluate [--main-test-only]
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from .. import config as C
from ..features import COLUMNS, clean, keep_columns


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def load_set(name):
    """Features (the 84 model columns) and labels of 'validation' or a test set such as 'test_6pts_0pct'."""
    F = np.load(C.PROCESSED / ('%s_features.npy' % name))
    y = np.load(C.PROCESSED / ('validation_labels.npy' if name == 'validation' else 'test_labels.npy'))
    return clean(F[:, [COLUMNS.index(c) for c in keep_columns()]]), np.asarray(y).astype(int)


def score(model, X, y):
    """Accuracy, log loss in bits, the confusion matrix and the predicted probabilities."""
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(1)]
    p_true = np.clip(proba[np.arange(len(y)), np.searchsorted(model.classes_, y)], 1e-15, 1)
    s = {'accuracy': (pred == y).mean(), 'logloss_bits': -np.log2(p_true).mean()}
    return s, confusion_matrix(y, pred, labels=range(20)), proba


def top2_recall(proba, y, classes):
    """Per mechanism: share of its experiments whose true mechanism is among the two most likely predictions."""
    top2 = classes[np.argsort(-proba, axis=1)[:, :2]]
    hit = (top2 == y[:, None]).any(1)
    return np.array([hit[y == m].mean() for m in range(20)])


def main():
    from .train import MODELS, NAMES, search_rows       # imported here to avoid a circular import
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--main-test-only', action='store_true', help='score only the main test set (6 points, 0%)')
    args = ap.parse_args()
    w = json.load(open(C.TABLES / 'winner.json'))
    tests = [C.MAIN_TEST] if args.main_test_only else ['test_%dpts_%dpct' % c for c in C.TEST_CONDITIONS]
    sets = {name: load_set(name) for name in ['validation'] + tests}

    y_search = np.load(C.PROCESSED / 'train_labels.npy')[search_rows()]
    dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(y_search), len(keep_columns())), np.float32), y_search)
    entries = [('Dummy (class prior)', dummy, False)]
    entries += [('%s (%s rows)' % (NAMES[m], format(C.N_SEARCH, ',')), C.MODELS / ('search_%s.joblib' % m), False)
                for m in MODELS if (C.MODELS / ('search_%s.joblib' % m)).exists()]
    entries.append(('%s, final (%s rows)' % (NAMES[w['model']], format(C.N_TRAIN_BATCHES * C.BATCH, ',')),
                    C.MODELS / ('final_%s.joblib' % w['model']), True))

    rows, recalls, top2 = [], [], []
    for label, model, is_final in entries:
        if not hasattr(model, 'predict_proba'):
            model = joblib.load(model)['model']
        for name, (X, y) in sets.items():
            s, cm, proba = score(model, X, y)
            rows.append({'model': label, 'set': name, **s})
            recalls += [{'model': label, 'set': name, 'mechanism': m, 'recall': r}
                        for m, r in zip(C.MECHS, cm.diagonal() / np.maximum(cm.sum(1), 1))]
            if is_final:
                pd.DataFrame(cm, index=C.MECHS, columns=C.MECHS).to_csv(C.TABLES / ('confusion_%s.csv' % name))
                top2 += [{'set': name, 'mechanism': m, 'recall': r, 'top2_recall': t}
                         for m, r, t in zip(C.MECHS, cm.diagonal() / np.maximum(cm.sum(1), 1),
                                            top2_recall(proba, y, model.classes_))]
            log('  %-44s %-16s accuracy %.4f  log loss %.3f bits' % (label, name, s['accuracy'], s['logloss_bits']))
    pd.DataFrame(rows).to_csv(C.TABLES / 'results.csv', index=False)
    pd.DataFrame(recalls).to_csv(C.TABLES / 'per_class_recall.csv', index=False)
    pd.DataFrame(top2).to_csv(C.TABLES / 'top2_recall.csv', index=False)


if __name__ == '__main__':
    main()
