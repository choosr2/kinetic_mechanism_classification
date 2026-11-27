"""Step 2: exploratory data analysis and the correlation filter (136 -> 84 features).

    1. data quality   missing, infinite and negative values and identical experiments in a random subset of
                      100,000 training experiments and in the test sets; class balance of every split; the
                      structure of every dataset file (report Appendix 1)
    2. features       on 200,000 random training experiments (after the authors' sampling): missing values per
                      feature, mutual information with the mechanism, Pearson correlation of the 136 features
    3. filter         while two features have Pearson |r| >= 0.9, drop the one with less mutual information with the
                      mechanism (ties: more missing values, then the later column) and recompute. Uses training data
                      only, before any model is fitted -> results/tables/drop_correlated.json
    4. distributions  robust summary of the 84 kept features: median, quartiles, 1st-99th percentiles, and the share
                      of extreme values (more than 3 IQR beyond the quartiles)
    5. mechanisms     median of every feature for each mechanism on the main test set

Outputs: results/tables/eda/ (CSV), results/tables/drop_correlated.json and
         results/report/Appendix_1_dataset_structure.csv.
Usage:  uv run python -m kinetic_mech.eda [--steps quality,features,mechanisms]
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .dataset import load_pickle, memmap_x2_train
from .features import COLUMNS, clean, families, keep_columns

OUT = C.TABLES / 'eda'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ------------------------------------------------------------------------------------------ 1. data quality
def data_quality(n_train=100_000, seed=C.SEED):
    """Missing / infinite / negative values and duplicates in a random training subset and the test sets."""
    y_all = np.asarray(load_pickle('y_train')).ravel()
    idx = np.sort(np.random.default_rng(seed).choice(len(y_all), n_train, replace=False))
    arrays = {'x1_train (subset)': np.asarray(load_pickle('x1_train'))[idx],
              'x2_train (subset)': np.array(memmap_x2_train(len(y_all))[idx]),
              'y_train (subset)': y_all[idx]}
    for pts, d in load_pickle('x2_test').items():
        for noise, a in d.items():
            arrays['x2_test, %d points, %d%% noise' % (pts, noise)] = a
    rows = []
    for name, a in arrays.items():
        a = np.asarray(a, dtype=float)
        rows.append({'array': name, 'shape': str(a.shape), 'NaN': int(np.isnan(a).sum()), 'inf': int(np.isinf(a).sum()),
                     'negative': int((a < 0).sum()), 'min': a.min(), 'max': a.max()})
    x2 = arrays['x2_train (subset)']
    flat = x2.reshape(len(x2), -1)
    duplicates = len(flat) - len(np.unique(flat, axis=0))
    Q = pd.DataFrame(rows)
    Q.attrs['duplicates'] = duplicates
    return Q, duplicates


CONTENTS = {'x1': 'Catalyst loading of each of the 4 runs',
            'x2': 'Kinetic curves: time, [S] and [P] for 4 runs at 21 time points',
            'y': 'Mechanism label (0–19 = M1–M20)'}
SPLIT_NAME = {'train': 'Training data', 'val': 'Validation data', 'test': 'Test data'}


def dataset_structure():
    """Report Appendix 1: every file of the dataset, its array shape and what it holds. x2_train (10 GB) is
    memory-mapped, so only its header is read."""
    rows = []
    for split in ('train', 'val', 'test'):
        n = len(np.asarray(load_pickle('y_' + split)).ravel())
        for part in ('x1', 'x2', 'y'):
            name = '%s_%s' % (part, split)
            if name == 'x2_train':
                shape, contents = str(memmap_x2_train(n).shape), CONTENTS[part]
            elif name == 'x2_test':
                sets = load_pickle(name)
                shapes = sorted({np.asarray(a).shape for d in sets.values() for a in d.values()}, key=lambda s: s[1])
                shape = 'Dictionary of %d arrays: %s' % (sum(len(d) for d in sets.values()),
                                                        ', '.join(str(s) for s in shapes))
                contents = ('Kinetic curves at %s time points (T) with %s%% noise; each array (%s, T+1, 12)'
                            % (' / '.join(str(p) for p in sorted(sets)),
                               ' / '.join(str(v) for v in sorted({k for d in sets.values() for k in d})),
                               format(n, ',')))
            else:
                shape, contents = str(np.asarray(load_pickle(name)).shape), CONTENTS[part]
            rows.append({'data_type': '%s (%s samples)' % (SPLIT_NAME[split], format(n, ',')), 'file': name,
                         'array_shape': shape, 'contents': contents})
    return pd.DataFrame(rows)


def class_balance():
    """Experiments per mechanism in each split."""
    out = {}
    for split in ('train', 'val', 'test'):
        y = np.asarray(load_pickle('y_' + split)).ravel().astype(int)
        out[split] = np.bincount(y, minlength=20)
    return pd.DataFrame(out, index=C.MECHS)


# ------------------------------------------------------------------------------------------ 2. features
def load_sample(feature_set='136'):
    """C.EDA_N random training experiments (seed C.EDA_SEED): features, mechanism, points per run and noise."""
    n_all = C.N_TRAIN_BATCHES * C.BATCH
    rows = np.sort(np.random.default_rng(C.EDA_SEED).choice(n_all, C.EDA_N, replace=False))
    F = np.load(C.PROCESSED / 'train_features.npy', mmap_mode='r')
    cols = keep_columns(feature_set)
    X = clean(np.asarray(F[rows])[:, [COLUMNS.index(c) for c in cols]])
    y = np.load(C.PROCESSED / 'train_labels.npy')[rows]
    draws = pd.read_csv(C.PROCESSED / 'train_draws.csv')
    block = rows // C.BATCH
    meta = pd.DataFrame({'mech': y.astype(int), 'n_pts': draws.n_pts.values[block], 'noise': draws.noise.values[block]})
    return pd.DataFrame(X, columns=cols), meta


def mutual_information(x, y, bins=C.MI_BINS, k=20):
    """Mutual information (bits) between one feature and the mechanism: quantile bins, missing values as a bin."""
    miss = np.isnan(x)
    v = x[~miss]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])) if len(v) else np.array([])
    b = np.zeros(len(x), dtype=int)
    b[~miss] = 1 + np.searchsorted(edges, v, side='right')
    joint = np.zeros((b.max() + 1, k))
    np.add.at(joint, (b, y), 1)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def feature_summary(F, meta):
    """Family, missing share and mutual information of every feature."""
    fam = families(F.columns)
    fam_of = {c: f for f, cs in fam.items() for c in cs}
    y = meta.mech.to_numpy()
    return pd.DataFrame({'family': [fam_of[c] for c in F.columns], 'feature': F.columns,
                         'missing_pct': [100 * F[c].isna().mean() for c in F.columns],
                         'mi_bits': [mutual_information(F[c].to_numpy(np.float64), y) for c in F.columns]})


def pearson(F):
    """Pearson correlation of the features, ordered by family (pairwise complete experiments)."""
    fam = families(F.columns)
    order = [c for f in C.FAMILY_ORDER for c in fam[f]]
    return F[order].corr(method='pearson')


def strong_pairs(R, cut=C.CORR_CUT):
    iu = np.triu_indices(len(R), 1)
    pairs = pd.DataFrame({'a': R.index[iu[0]], 'b': R.columns[iu[1]], 'r': R.to_numpy()[iu]})
    return pairs[pairs.r.abs() >= cut].sort_values('r', key=abs, ascending=False)


# ------------------------------------------------------------------------------------------ 3. filter
def correlation_filter(R, summary, cut=C.CORR_CUT):
    """Drop one feature of the most correlated pair (the one with less mutual information; ties: more missing, then
    the later column) and repeat until no pair has |r| >= cut."""
    order = list(R.index)
    A = R.abs()
    mi, miss = summary.set_index('feature').mi_bits, summary.set_index('feature').missing_pct
    fam_of = summary.set_index('feature').family
    keep, dropped = list(order), []
    better = lambda c: (mi[c], -miss[c], -order.index(c))          # noqa: E731  larger tuple = kept
    while True:
        sub = np.nan_to_num(A.loc[keep, keep].to_numpy())
        np.fill_diagonal(sub, 0)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] < cut:
            break
        a, b = keep[i], keep[j]
        lose, win = (a, b) if better(a) < better(b) else (b, a)
        dropped.append({'dropped': lose, 'family': fam_of[lose], 'kept_partner': win, 'partner_family': fam_of[win],
                        'abs_r': round(float(sub[i, j]), 4), 'mi_dropped': float(mi[lose]), 'mi_kept': float(mi[win])})
        keep.remove(lose)
    left = A.loc[keep, keep].to_numpy().copy()
    np.fill_diagonal(left, np.nan)
    return {'rule': 'Pearson |r| >= %.1f on %s training experiments; iteratively drop the member of the most correlated '
                    'pair with less mutual information with the mechanism (ties: more missing, then later column)'
                    % (cut, format(C.EDA_N, ',')),
            'cutoff': cut, 'n_start': len(order), 'n_dropped': len(dropped), 'n_kept': len(keep),
            'max_abs_r_left': round(float(np.nanmax(left)), 4),
            'dropped': [d['dropped'] for d in dropped], 'kept': keep, 'details': dropped}


# ------------------------------------------------------------------------------------------ 4. distributions
def robust_scale(v):
    """Quantiles and a robust scale: the IQR, or (if it is zero) the IQR-equivalent of the 10-90% range, or of the
    standard deviation."""
    q = np.quantile(v, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    iqr, scale_from = q[4] - q[2], 'IQR'
    if iqr <= 1e-12 * (abs(q[3]) + 1):
        iqr, scale_from = (q[5] - q[1]) / 2.5631 * 1.349, '10-90% range'
        if iqr <= 1e-12 * (abs(q[3]) + 1):
            iqr, scale_from = 1.349 * v.std(), 'standard deviation'
    return q, iqr, scale_from


def distributions(F):
    """Per feature: missing share, median, quartiles, 1st/99th percentiles, min, max, robust scale and the share of
    extreme values (beyond Q1 - 3 IQR or Q3 + 3 IQR)."""
    fam = families(F.columns)
    fam_of = {c: f for f, cs in fam.items() for c in cs}
    rows = []
    for c in F.columns:
        x = F[c].to_numpy(np.float64)
        ok = ~np.isnan(x)
        v = x[ok]
        q, iqr, scale_from = robust_scale(v)
        ext = ok & ((x < q[2] - C.FENCE * iqr) | (x > q[4] + C.FENCE * iqr))
        rows.append({'family': fam_of[c], 'feature': c, 'missing_pct': 100 * (~ok).mean(), 'median': q[3],
                     'q1': q[2], 'q3': q[4], 'p1': q[0], 'p99': q[6], 'min': v.min(), 'max': v.max(), 'scale': iqr,
                     'scale_from': scale_from, 'extreme_pct': 100 * ext.sum() / ok.sum(), 'n_extreme': int(ext.sum()),
                     'n_values': int(ok.sum())})
    D = pd.DataFrame(rows)
    overall = 100 * D.n_extreme.sum() / D.n_values.sum()
    return D, overall


# ------------------------------------------------------------------------------------------ main
def step_quality():
    """Data quality and class balance (needs only data/raw)."""
    OUT.mkdir(parents=True, exist_ok=True)
    Q, dup = data_quality()
    Q['identical_experiments'] = [dup if a == 'x2_train (subset)' else np.nan for a in Q.array]
    Q.to_csv(OUT / 'data_quality.csv', index=False)
    log('data quality (100,000 training experiments and the test sets); identical training experiments: %d\n%s'
        % (dup, Q.to_string(index=False)))
    B = class_balance()
    B.to_csv(OUT / 'class_balance.csv')
    log('class balance: every mechanism %.1f-%.1f%% of each split'
        % (100 * (B / B.sum()).min().min(), 100 * (B / B.sum()).max().max()))
    C.REPORT.mkdir(parents=True, exist_ok=True)
    S = dataset_structure()
    S.to_csv(C.REPORT / 'Appendix_1_dataset_structure.csv', index=False)
    log('report Appendix 1 (dataset structure):\n%s' % S.to_string(index=False))


def step_mechanisms():
    """Median of every feature for each mechanism on the main test set (6 points, 0% noise), and the share of
    experiments where it could be computed. Supports the report's statements about what a family measures for a
    mechanism (e.g. the VTNA order of M8, the inhibitor amount of M20)."""
    F = np.load(C.PROCESSED / ('%s_features.npy' % C.MAIN_TEST))
    y = np.load(C.PROCESSED / 'test_labels.npy').astype(int)
    X = pd.DataFrame(clean(F), columns=COLUMNS)
    fam = families(COLUMNS)
    fam_of = {c: f for f, cs in fam.items() for c in cs}
    kept = set(keep_columns('84'))
    med = X.groupby(y).median().T
    med.columns = C.MECHS
    have = X.notna().groupby(y).mean().T
    out = pd.concat([pd.DataFrame({'family': [fam_of[c] for c in COLUMNS], 'feature': COLUMNS,
                                   'in_model': [c in kept for c in COLUMNS]}, index=COLUMNS),
                     med.add_prefix('median_'), (100 * have).add_prefix('computed_pct_')], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(OUT / 'feature_medians_by_mechanism_test.csv', index=False)
    log('feature medians by mechanism (%s, %d experiments): %s' % (C.MAIN_TEST, len(y), out.shape))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', default='quality,features,mechanisms',
                    help='quality (data/raw only), features and mechanisms (data/processed)')
    steps = ap.parse_args().steps.split(',')
    if 'quality' in steps:
        step_quality()
    if 'features' in steps:
        step_features()
    if 'mechanisms' in steps:
        step_mechanisms()


def step_features():
    """Missing values, mutual information, Pearson correlation, the correlation filter and distributions."""
    OUT.mkdir(parents=True, exist_ok=True)
    F, meta = load_sample('136')
    S = feature_summary(F, meta)
    S.round(4).to_csv(OUT / 'feature_summary_136.csv', index=False)
    R = pearson(F)
    R.round(4).to_csv(OUT / 'pearson_136.csv')
    P = strong_pairs(R)
    P.round(4).to_csv(OUT / 'pairs_abs_r_ge_0.9.csv', index=False)
    log('136 features: %.2f%% of values missing; %d pairs with |r| >= %.1f'
        % (100 * F.isna().to_numpy().mean(), len(P), C.CORR_CUT))

    drop = correlation_filter(R, S)
    with open(C.TABLES / 'drop_correlated.json', 'w') as f:
        json.dump(drop, f, indent=1)
    log('correlation filter: dropped %d, kept %d (largest |r| left %.3f)'
        % (drop['n_dropped'], drop['n_kept'], drop['max_abs_r_left']))

    F84 = F[keep_columns('84')]
    D, overall = distributions(F84)
    D.round(4).to_csv(OUT / 'distribution_84.csv', index=False)
    log('84 features: %.2f%% of values are extreme (more than %.0f IQR beyond the quartiles)' % (overall, C.FENCE))


if __name__ == '__main__':
    main()
