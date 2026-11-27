"""Step 5: grouped permutation importance of the final model (no retraining).

For each of the 12 feature families, and each of the 3 blocks of related families (graphical tests, fitted laws,
mass balance), the unit's columns are permuted together with one random row permutation of the test set, so the
correlations inside the unit are kept, and the test set is predicted again. This is repeated 3 times per unit.
Permutation seeds: default_rng([31, k, r]) for unit k (families 1-12 in config.FAMILY_RULES order, blocks 13-15)
and repeat r.

Per unit, from the mean of the repeats (all in percentage points):
    accuracy change              mean change in test accuracy
    D[i, j]                      change in the share of true-i experiments predicted as j (diagonal = recall change)
    significant                  |change| >= 2 points for recall, >= 1 point for other cells, and > 3 standard errors
                                 of the repeat mean (the unpermuted model is deterministic, so it adds no noise)
    mechanisms affected          significant recall losses of >= 5 points
    role                         Minor / Specialist / Group specialist / Generalist / Intermediate, from how the
                                 significant recall losses are spread over the mechanisms (config.py)
Outputs (results/tables/permutation/): confusions.npz, runs.csv, recall_change.csv, confusion_shift.csv,
family_summary.csv

Noisy test sets (--noisy): the noise-and-sampling family is also permuted, with the same seeds, on the test sets with
1% and 5% noise (the same 100,000 experiments, so the rows are reordered exactly as on the noise-free set)
-> results/tables/permutation_noisy/summary.csv, recall_change.csv
Usage:  uv run python -m kinetic_mech.permutation [--analyze-only] [--noisy]
"""
import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import families, keep_columns
from .modeling.evaluate import load_set

OUT = C.TABLES / 'permutation'
FAMILY_NAMES = [n for n, _ in C.FAMILY_RULES]
BLOCK_UNITS = {'block ' + g: members for g, members in C.BLOCKS.items()}
UNITS = FAMILY_NAMES + list(BLOCK_UNITS)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def key(unit, r):
    return 'hard__%s__%d' % (slug(unit), r)


def seed_index(unit):
    """k of the permutation seed: families 1-12, blocks 13-15."""
    if unit in BLOCK_UNITS:
        return 1 + len(FAMILY_NAMES) + list(BLOCK_UNITS).index(unit)
    return 1 + FAMILY_NAMES.index(unit)


def unit_columns(unit, cols):
    fam = families(cols)
    return [cols.index(c) for f in BLOCK_UNITS.get(unit, [unit]) for c in fam[f]]


def permuted_confusion(model, X, y, jj, seed):
    """Confusion matrix (counts) after permuting columns jj together; jj = [] gives the unpermuted model."""
    Xs = X
    if jj:
        perm = np.random.default_rng(seed).permutation(len(y))
        Xs = X.copy()
        Xs[:, jj] = X[perm][:, jj]
    pred = model.predict_proba(Xs).argmax(1)
    hard = np.zeros((20, 20))
    np.add.at(hard, (y, pred), 1)
    return hard


def run(set_name=C.MAIN_TEST, units=None, repeats=range(C.N_REPEATS)):
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(C.MODELS / 'final_lightgbm.joblib')['model']
    X, y = load_set(set_name)
    cols = keep_columns()
    npz = OUT / 'confusions.npz'
    Z = dict(np.load(npz)) if npz.exists() else {}
    runs = pd.read_csv(OUT / 'runs.csv').to_dict('records') if (OUT / 'runs.csv').exists() else []
    jobs = [('full', 0, [], 0)] + [(u, seed_index(u), unit_columns(u, cols), r) for u in (units or UNITS) for r in repeats]
    for unit, k, jj, r in jobs:
        if key(unit, r) in Z:
            continue
        hard = permuted_confusion(model, X, y, jj, [C.PERM_SEED, k, r])
        Z[key(unit, r)] = hard
        np.savez_compressed(npz, **Z)
        runs.append({'unit': unit, 'repeat': r + 1, 'k': k, 'n_columns': len(jj), 'accuracy': np.trace(hard) / len(y)})
        pd.DataFrame(runs).to_csv(OUT / 'runs.csv', index=False)
        log('%-28s repeat %d: %2d columns permuted, accuracy %.4f' % (unit, r + 1, len(jj), runs[-1]['accuracy']))


CATEGORY = np.concatenate([[name] * (b - a) for name, a, b in C.CATEGORIES])     # category of M1-M20


def role(loss):
    """Role of a unit from its significant recall losses L (points) over the 20 mechanisms (see config.py)."""
    total = loss.sum()
    if total <= 0:
        return {'role': 'Minor', 'breadth': 0.0, 'top_category': '', 'top_category_share': 0.0}
    breadth = total ** 2 / (loss ** 2).sum()
    peak = loss.max()
    share = pd.Series(loss).groupby(CATEGORY).sum() / total
    top, top_share = share.idxmax(), share.max()
    if peak < C.PEAK_MINOR:
        r = 'Minor'
    elif breadth >= C.BREADTH_SPECIALIST and top_share >= C.GROUP_SHARE:
        r = 'Group specialist'
    elif breadth >= C.BREADTH_GENERALIST:
        r = 'Generalist'
    elif breadth < C.BREADTH_SPECIALIST and peak >= C.PEAK_SPECIALIST:
        r = 'Specialist'
    else:
        r = 'Intermediate'
    return {'role': r, 'breadth': breadth, 'top_category': top, 'top_category_share': top_share}


def noisy(unit='Noise and sampling', sets=C.NOISY_TESTS):
    """Permute one unit on the noisy test sets (same seeds and repeats as on the noise-free set)."""
    out = C.TABLES / 'permutation_noisy'
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(C.MODELS / 'final_lightgbm.joblib')['model']
    cols = keep_columns()
    jj, k = unit_columns(unit, cols), seed_index(unit)
    summary, recalls = [], []
    for set_name in sets:
        X, y = load_set(set_name)
        full = permuted_confusion(model, X, y, [], None)
        n_true = full.sum(1)
        S = np.stack([100 * permuted_confusion(model, X, y, jj, [C.PERM_SEED, k, r]) / n_true[:, None]
                      for r in range(C.N_REPEATS)])
        F = 100 * full / n_true[:, None]
        acc = np.array([np.trace(s * n_true[:, None] / 100) / len(y) for s in S]) * 100
        d, se = np.diag(S.mean(0) - F), np.diag(S.std(0, ddof=1)) / np.sqrt(len(S))
        sig = (np.abs(d) >= C.MIN_RECALL) & (np.abs(d) > C.N_SE * se)
        summary.append({'set': set_name, 'unit': unit, 'n_columns': len(jj),
                        'accuracy_before': 100 * np.trace(full) / len(y), 'accuracy_after': acc.mean(),
                        'loss_points': 100 * np.trace(full) / len(y) - acc.mean(), 'loss_se': acc.std(ddof=1) / np.sqrt(len(acc)),
                        'mechanisms_affected': int((sig & (d <= -C.AFFECTS)).sum())})
        recalls += [{'set': set_name, 'unit': unit, 'mech': m, 'recall_before': F[i, i], 'recall_after': F[i, i] + d[i],
                     'change_points': d[i], 'se_points': se[i], 'significant': bool(sig[i])}
                    for i, m in enumerate(C.MECHS)]
        log('%s, %s: accuracy loss %.2f points' % (set_name, unit, summary[-1]['loss_points']))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(recalls).to_csv(out / 'recall_change.csv', index=False)


def analyze():
    Z = dict(np.load(OUT / 'confusions.npz'))
    runs = pd.read_csv(OUT / 'runs.csv')
    full = Z[key('full', 0)]
    n_true = full.sum(1)
    F = 100 * full / n_true[:, None]
    acc0 = 100 * np.trace(full) / full.sum()
    rec_rows, shift_rows, summary = [], [], []
    for u in [u for u in UNITS if key(u, 0) in Z]:
        S = np.stack([100 * Z[key(u, r)] / n_true[:, None] for r in range(C.N_REPEATS) if key(u, r) in Z])
        d = S.mean(0) - F
        se = S.std(0, ddof=1) / np.sqrt(len(S)) if len(S) > 1 else np.zeros_like(d)
        for i, m in enumerate(C.MECHS):
            sig = abs(d[i, i]) >= C.MIN_RECALL and abs(d[i, i]) > C.N_SE * se[i, i]
            rec_rows.append({'unit': u, 'mech': m, 'recall_before': F[i, i], 'recall_after': F[i, i] + d[i, i],
                             'change_points': d[i, i], 'se_points': se[i, i], 'significant': sig})
            for j, mj in enumerate(C.MECHS):
                if i != j:
                    sig = abs(d[i, j]) >= C.MIN_CELL and abs(d[i, j]) > C.N_SE * se[i, j]
                    shift_rows.append({'unit': u, 'true': m, 'predicted_as': mj, 'before_points': F[i, j],
                                       'change_points': d[i, j], 'se_points': se[i, j], 'significant': sig})
        loss = np.array([-d[i, i] if (d[i, i] <= -C.MIN_RECALL and -d[i, i] > C.N_SE * se[i, i]) else 0.0
                         for i in range(20)])
        acc = runs[runs.unit == u].accuracy
        summary.append({'unit': u, 'kind': 'block' if u in BLOCK_UNITS else 'family',
                        'n_columns': int(runs[runs.unit == u].n_columns.iloc[0]),
                        'accuracy_change_points': 100 * acc.mean() - acc0,
                        'accuracy_change_sd_points': 100 * acc.std(ddof=1),
                        'mechanisms_affected': int((loss >= C.AFFECTS).sum()),
                        'largest_loss_points': loss.max(),
                        'largest_loss_mech': C.MECHS[int(loss.argmax())] if loss.max() > 0 else '',
                        **role(loss)})
    pd.DataFrame(rec_rows).to_csv(OUT / 'recall_change.csv', index=False)
    pd.DataFrame(shift_rows).to_csv(OUT / 'confusion_shift.csv', index=False)
    S = pd.DataFrame(summary).sort_values('accuracy_change_points')
    S.to_csv(OUT / 'family_summary.csv', index=False)
    log('unpermuted accuracy %.2f%%\n%s' % (acc0, S.round(2).to_string(index=False)))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analyze-only', action='store_true', help='recompute the tables from confusions.npz')
    ap.add_argument('--noisy', action='store_true', help='also permute the noise family on the noisy test sets')
    args = ap.parse_args()
    if not args.analyze_only:
        run()
    analyze()
    if args.noisy:
        noisy()


if __name__ == '__main__':
    main()
