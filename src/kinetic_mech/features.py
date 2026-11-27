"""The 136 engineered kinetic features, one function per feature family.

Every function takes the sorted runs of a batch of experiments (all with the same number of time points) and returns
a dict of columns, one value per experiment. compute_features() puts the 136 columns together in a fixed order (the
order the models were trained with). A feature is NaN when its test cannot be computed for that experiment (for
example a run that never reaches the conversion the test needs); the models handle NaN directly.

Notation: t, S, P are time, substrate and product of one run, shape (n experiments, points). X = (1 - S + P) / 2 is
the pooled conversion, which averages S and P and so halves the noise variance. c = [cat]0 of the run.
Runs 1-3 differ only in catalyst loading (rising from run 1 to run 3); run 4 is the same-excess run.
What each feature measures is in report Appendix 6 (results/report/); the source paper of each method is in the
supporting document (results/supporting/).
"""
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import expi

from . import config as C

PAIRS = [(0, 1), (1, 2), (0, 2)]          # run pairs 1-2, 2-3, 1-3
VTNA_LEVELS = 8                           # conversion levels used by the pooled order (VTNA)
LOSS_LAWS = ('free', 'sub', 'sub2', 'pfree', 'psub')   # amount of catalyst lost does not depend on loading
SHARE_LAWS = ('free', 'sub')                           # fraction lost does not depend on loading (the _b laws)
INHIBITOR_SHAPES = (0.5, 2.0, 8.0)        # shape g of the used-up-inhibitor laws

# the 136 columns, in the order the models read them
COLUMNS = [
    'P_end_1', 't_end_1', 'shape25_1', 'shape50_1', 't_half_frac_1', 'P_end_2', 't_end_2', 'shape25_2', 'shape50_2',
    't_half_frac_2', 'P_end_3', 't_end_3', 'shape25_3', 'shape50_3', 't_half_frac_3', 'P_end_4', 't_end_4',
    'shape25_4', 'shape50_4', 't_half_frac_4', 'cat_order_12', 'cat_order_23', 'cat_order_13', 'same_excess_log_ratio',
    'cat0_1', 'cat0_2', 'cat0_3', 'cat_spread', 'X_P_end_1', 'X_shape25_1', 'X_shape50_1', 'X_t_half_frac_1',
    'X_P_end_2', 'X_shape25_2', 'X_shape50_2', 'X_t_half_frac_2', 'X_P_end_3', 'X_shape25_3', 'X_shape50_3',
    'X_t_half_frac_3', 'X_P_end_4', 'X_shape25_4', 'X_shape50_4', 'X_t_half_frac_4', 'X_cat_order_12',
    'X_cat_order_23', 'X_cat_order_13', 'X_same_excess_log_ratio', 'noise_est', 'max_gap_1', 'max_gap_2', 'max_gap_3',
    'max_gap_4', 'deficit_slope', 'deficit_per_cat_1', 'deficit_per_cat_2', 'deficit_per_cat_3', 'deficit_per_cat_4',
    'selwyn_12', 'selwyn_23', 'selwyn_13', 'selwyn_rms', 'same_excess_mean', 'same_excess_gap_5_20',
    'same_excess_gap_20_35', 'vtna_order_all', 'vtna_rms_all', 'vtna_rms_noise_all', 'vtna_width_all',
    'vtna_res_low_all', 'vtna_res_high_all', 'vtna_order_core', 'vtna_rms_core', 'vtna_rms_noise_core',
    'vtna_width_core', 'vtna_res_low_core', 'vtna_res_high_core', 'mm_km', 'mm_sat', 'mm_a', 'mm_b', 'mm_rmse',
    'mm_res_1', 'mm_res_2', 'mm_res_3', 'mm_res_4', 'law_rel_free', 'law_loss_free', 'law_res4_free', 'law_rel_sub',
    'law_loss_sub', 'law_res4_sub', 'law_rel_sub2', 'law_loss_sub2', 'law_res4_sub2', 'law_rel_pfree',
    'law_loss_pfree', 'law_res4_pfree', 'law_rel_psub', 'law_loss_psub', 'law_res4_psub', 'law_rel_free_b',
    'law_loss_free_b', 'law_res4_free_b', 'law_rel_sub_b', 'law_loss_sub_b', 'law_res4_sub_b', 'law_best',
    'inh_gain_poison', 'inh_amount_poison', 'inh_res4_poison', 'inh_gain_free_0.5', 'inh_gain_free_2',
    'inh_gain_free_8', 'inh_best_free', 'inh_amount_free', 'inh_res4_free', 'inh_gain_sub_0.5', 'inh_gain_sub_2',
    'inh_gain_sub_8', 'inh_best_sub', 'inh_amount_sub', 'inh_res4_sub', 'inh_vs_law', 'dk_km1', 'dk_km2', 'dk_eps',
    'dk_rmse', 'dk_res_4', 'dk_km1_early', 'dk_km1_late', 'dk_gap1', 'dk_gap2', 'dk_logratio', 'dk_sat', 'dk_sat_gap']


# ============================================================================================ helpers
def sorted_runs(x2):
    """(t, S, P) of each of the 4 runs, every array (n, points) and sorted by time."""
    order = np.argsort(x2[:, :, [0, 3, 6, 9]], axis=1, kind='stable')
    return [tuple(np.take_along_axis(x2[:, :, 3 * r + c], order[:, :, r], axis=1) for c in range(3))
            for r in range(4)]


def pooled(run):
    """Pooled conversion X = (1 - S + P) / 2 of one run."""
    t, S, P = run
    return (1 - S + P) / 2


def interp_rows(t, y, q):
    """Row-wise linear interpolation y(q), one query per row; t sorted in each row."""
    n, m = t.shape
    rows = np.arange(n)
    k = np.clip((t <= q[:, None]).sum(1) - 1, 0, m - 2)
    t0, t1, y0, y1 = t[rows, k], t[rows, k + 1], y[rows, k], y[rows, k + 1]
    dt = t1 - t0
    w = np.clip(np.divide(q - t0, dt, out=np.zeros(n), where=dt > 0), 0, 1)
    return y0 + w * (y1 - y0)


def interp_many(t, y, Q):
    """Row-wise y(Q) for a matrix of query times Q (n, q); NaN queries stay NaN."""
    out = np.full(Q.shape, np.nan)
    for j in range(Q.shape[1]):
        ok = ~np.isnan(Q[:, j])
        if ok.any():
            out[ok, j] = interp_rows(t[ok], y[ok], Q[ok, j])
    return out


def cross_time(t, y, level):
    """Row-wise first time at which a rising curve y reaches `level` (linear between points; NaN if never)."""
    n = len(t)
    rows = np.arange(n)
    hit = y >= level[:, None]
    k = hit.argmax(1)
    km = np.maximum(k, 1)
    y0, y1, t0, t1 = y[rows, km - 1], y[rows, km], t[rows, km - 1], t[rows, km]
    dy = y1 - y0
    w = np.clip(np.divide(level - y0, dy, out=np.zeros(n), where=dy > 0), 0, 1)
    out = t0 + w * (t1 - t0)
    out[k == 0] = t[k == 0, 0]
    out[~hit.any(1)] = np.nan
    return out


def monotone(y):
    """Monotone envelope of a rising curve, so that noise cannot create false crossings."""
    return 0.5 * (np.maximum.accumulate(y, axis=1) + np.minimum.accumulate(y[:, ::-1], axis=1)[:, ::-1])


def safe_log_ratio(a, b):
    ok = (a > 0) & (b > 0)
    return np.where(ok, np.log(np.where(ok, a, 1) / np.where(ok, b, 1)), np.nan)


def safe_div(a, b, where):
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=where)


def nanmean_rows(a):
    ok = ~np.isnan(a)
    n = ok.sum(1)
    return np.divide(np.where(ok, a, 0).sum(1), n, out=np.full(len(a), np.nan), where=n > 0)


def matching_run(x1):
    """Index (0-2) of the run among 1-3 with the same loading as run 4, shape (n, 1)."""
    return np.abs(x1[:, :3] - x1[:, 3:4]).argmin(1)[:, None]


# ============================================================================================ graphical tests
def catalyst_order(x1, runs):
    """Catalyst order from pairs of runs (Burés 2016): how much faster the higher-loading run makes 0.2 of product,
    order = -ln(t_b / t_a) / ln(c_b / c_a). Also from pooled conversion X (X_ prefix)."""
    f = {}
    for prefix, curve in (('', lambda run: run[2]), ('X_', pooled)):
        t02 = []
        for run in runs[:3]:
            y = curve(run)
            t02.append(cross_time(run[0], y, y[:, 0] + 0.2))
        for a, b in PAIRS:
            lc = np.log(x1[:, b] / x1[:, a])
            order = -safe_log_ratio(t02[b], t02[a]) / np.where(np.abs(lc) > 0.1, lc, np.nan)
            f['%scat_order_%d%d' % (prefix, a + 1, b + 1)] = np.clip(order, -3, 5)
    return f


def same_excess(x1, runs):
    """Same-excess test (Blackmond 2005): run 4 starts where the matching run (same loading) is part-way through.
    Shift the matching run in time so both start at run 4's level and compare them; a mismatch means the catalyst
    changed during the matching run (lost: run 4 is faster; activated: run 4 is slower)."""
    f = {}
    m = matching_run(x1)
    # time to make the first 0.1 of product (or of X) after run 4's start: run 4 against the matching run
    for prefix, curve in (('', lambda run: run[2]), ('X_', pooled)):
        y4 = curve(runs[3])
        t_m = np.choose(m, [runs[i][0] for i in range(3)])
        y_m = np.choose(m, [curve(runs[i]) for i in range(3)])
        start = y4[:, 0]
        seg_m = cross_time(t_m, y_m, start + 0.1) - cross_time(t_m, y_m, start)
        seg_4 = cross_time(runs[3][0], y4, start + 0.1)
        f['%ssame_excess_log_ratio' % prefix] = safe_log_ratio(seg_4, seg_m)

    # whole-run gap in X after the shift, and time ratios over two conversion windows
    t = [r[0] for r in runs]
    X = [pooled(r) for r in runs]
    Xm = [monotone(v) for v in X]
    t_par, X_par, Xm_par = (np.choose(m, v[:3]) for v in (t, X, Xm))
    t4, X4, Xm4 = t[3], X[3], Xm[3]
    p0 = X4[:, 0]
    shift = cross_time(t_par, Xm_par, p0)
    Q = t4[:, 1:] + shift[:, None]
    Q[Q > t_par[:, -1:]] = np.nan
    gap = interp_many(t_par, X_par, Q) - np.where(np.isnan(Q), np.nan, X4[:, 1:])   # < 0: run 4 is ahead
    f['same_excess_mean'] = nanmean_rows(gap)
    for lo, hi in ((0.05, 0.2), (0.2, 0.35)):           # both start after the initial catalyst-binding burst
        dt_par = cross_time(t_par, Xm_par, p0 + hi) - cross_time(t_par, Xm_par, p0 + lo)
        dt_4 = cross_time(t4, Xm4, p0 + hi) - cross_time(t4, Xm4, p0 + lo)
        f['same_excess_gap_%d_%d' % (100 * lo, 100 * hi)] = safe_log_ratio(dt_par, dt_4)
    return f


def selwyn(x1, runs):
    """Selwyn test (Selwyn 1965): plotted against [cat]0 * t, runs 1-3 overlap if the catalyst is stable and first
    order. selwyn_ab = mean gap in X between run a and run b at run b's points; selwyn_rms = spread of all three."""
    t = [r[0] for r in runs]
    X = [pooled(r) for r in runs]
    tau = [t[r] * x1[:, r:r + 1] for r in range(3)]
    f = {}
    for a, b in PAIRS:
        end = np.minimum(tau[a][:, -1], tau[b][:, -1])
        Q = tau[b][:, 1:].copy()
        Q[(Q > end[:, None]) | (Q <= 0)] = np.nan
        high = np.where(np.isnan(Q), np.nan, X[b][:, 1:])
        f['selwyn_%d%d' % (a + 1, b + 1)] = nanmean_rows(interp_many(tau[a], X[a], Q) - high)
    end = np.minimum.reduce([tau[r][:, -1] for r in range(3)])
    Q = tau[2][:, 1:].copy()
    Q[(Q > end[:, None]) | (Q <= 0)] = np.nan
    curves = np.stack([interp_many(tau[0], X[0], Q), interp_many(tau[1], X[1], Q),
                       np.where(np.isnan(Q), np.nan, X[2][:, 1:])], axis=2)
    f['selwyn_rms'] = np.sqrt(nanmean_rows(curves.var(axis=2)))
    return f


def vtna(x1, runs, noise):
    """Pooled catalyst order (VTNA; Burés 2016, Auto-VTNA 2024). Scaling time by [cat]0^n lines runs 1-3 up when n
    is the order in catalyst. At 8 conversion levels, ln t = f(X) - n ln c is fitted across the runs:
    vtna_order = n; vtna_rms = remaining misfit; vtna_rms_noise = misfit / noise_est; vtna_width = range of orders
    within 15% of the best error; vtna_res_low / _high = mean residual of run 1 / run 3.
    _all: from 5% conversion up to the highest level all three runs reach; _core: 15-85% only."""
    n = len(x1)
    t = [r[0] for r in runs[:3]]
    Xm = [monotone(pooled(r)) for r in runs[:3]]
    x = np.log(x1[:, :3])
    x = x - x.mean(1, keepdims=True)
    sxx = (x ** 2).sum(1)
    top = np.min([v.max(1) for v in Xm], axis=0)
    f = {}
    for tag, lo, hi in (('all', np.full(n, 0.05), top), ('core', np.full(n, 0.15), np.minimum(top, 0.85))):
        grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, VTNA_LEVELS)[None, :]
        T = np.stack([np.stack([cross_time(t[r], Xm[r], grid[:, j]) for j in range(VTNA_LEVELS)], 1)
                      for r in range(3)], 1)
        T[hi - lo < 0.05] = np.nan
        y = np.log(np.where(T > 0, T, np.nan))
        yd = y - y.mean(1, keepdims=True)                     # removes f(X) at each level
        m = (~np.isnan(yd).any(1)).sum(1)
        ok = (m > 0) & (sxx > 1e-6)
        slope = safe_div(np.nansum((x[:, :, None] * yd).sum(1), axis=1), m * sxx, ok)
        e = yd - slope[:, None, None] * x[:, :, None]
        rss = np.nansum(e ** 2, axis=(1, 2))
        rms = np.sqrt(safe_div(rss, 2.0 * m - 1, ok))
        f['vtna_order_' + tag] = np.clip(-slope, -3, 5)
        f['vtna_rms_' + tag] = rms
        f['vtna_rms_noise_' + tag] = rms / np.maximum(noise, 1e-4)
        f['vtna_width_' + tag] = np.clip(2 * np.sqrt(safe_div(0.15 * rss, m * sxx, ok)), 0, 5)
        f['vtna_res_low_' + tag] = nanmean_rows(e[:, 0])
        f['vtna_res_high_' + tag] = nanmean_rows(e[:, 2])
    return f


# ============================================================================================ integrated rate-law fits
def mm_fit(x1, runs):
    """Integrated Michaelis-Menten law, c*t = a(S0 - S) + b ln(S0/S), fitted by least squares to runs 1-3 together
    (Schnell & Mendoza 1997; Goudar et al. 1999); run 4 is held out. mm_km = b/a (a Km-like shape number),
    mm_sat = share of the time to half conversion from the zero-order term, mm_a / mm_b = coefficients,
    mm_rmse = misfit, mm_res_r = mean residual of run r. Residuals are scaled by the mean c*t."""
    u1, u2, y, ok = [], [], [], []
    for r in range(4):
        t, X = runs[r][0], pooled(runs[r])
        x0 = X[:, :1]
        left = 1 - X[:, 1:]
        good = (left > 0.05) & (X[:, 1:] - x0 > 0.005)
        u1.append(np.where(good, X[:, 1:] - x0, 0.0))
        u2.append(np.where(good, np.log((1 - x0) / np.where(good, left, 1)), 0.0))
        y.append(np.where(good, x1[:, r:r + 1] * t[:, 1:], 0.0))
        ok.append(good)
    U1, U2, Y, W = (np.concatenate(v[:3], axis=1) for v in (u1, u2, y, ok))
    s11, s12, s22 = (U1 * U1).sum(1), (U1 * U2).sum(1), (U2 * U2).sum(1)
    b1, b2 = (U1 * Y).sum(1), (U2 * Y).sum(1)
    det = s11 * s22 - s12 ** 2
    count = W.sum(1)
    good = (det > 1e-10 * s11 * s22) & (count >= 3)
    a = safe_div(s22 * b1 - s12 * b2, det, good)
    b = safe_div(s11 * b2 - s12 * b1, det, good)
    ref = safe_div(Y.sum(1), count.astype(float), count > 0)
    res = [np.where(ok[r], y[r] - a[:, None] * u1[r] - b[:, None] * u2[r], np.nan) for r in range(4)]
    rss = np.nansum(np.concatenate(res[:3], axis=1) ** 2, axis=1)
    f = {'mm_km': np.clip(safe_div(b, a, good & (a != 0)), -10, 50),
         'mm_sat': np.clip(safe_div(0.5 * a, 0.5 * a + np.log(2) * b, good & (0.5 * a + np.log(2) * b != 0)), -1, 2),
         'mm_a': a / ref, 'mm_b': b / ref,
         'mm_rmse': np.sqrt(safe_div(rss, count - 2.0, count > 2)) / ref}
    for r in range(4):
        f['mm_res_%d' % (r + 1)] = nanmean_rows(res[r]) / ref
    return f


def law_parts(x1, runs):
    """Pooled substrate left and the fit target c*t of each run, with unusable points zeroed."""
    parts = []
    for r in range(4):
        t, S, P = runs[r]
        X = (1 - S + P) / 2
        x0 = X[:, :1]
        s0 = 1 - x0                                       # equals S0 in run 4, because S0 + P0 = 1
        s = 1 - X[:, 1:]
        good = (s > 0.05) & (X[:, 1:] - x0 > 0.005)
        s = np.where(good, s, s0)                         # every column is then 0 at unusable points
        c = x1[:, r:r + 1]
        parts.append({'t': np.where(good, t[:, 1:], 0.0), 's': s, 's0': s0, 'P0': P[:, :1], 'e': s0 - s,
                      'L': np.log(s0 / s), 'good': good, 'c': c, 'y': np.where(good, c * t[:, 1:], 0.0)})
    return parts


def fit_design(parts, extra=()):
    """Least squares of c*t on (S0 - S, ln S0/S, extra terms) over runs 1-3 of each experiment; run 4 held out.
    Returns the coefficients, the residual sum of squares on runs 1-3 and the residuals of every run."""
    D = [np.stack([p['e'], p['L']] + [z[r] for z in extra], axis=2) for r, p in enumerate(parts)]
    A = np.concatenate(D[:3], axis=1)
    Y = np.concatenate([p['y'] for p in parts[:3]], axis=1)
    count = np.concatenate([p['good'] for p in parts[:3]], axis=1).sum(1)
    k = A.shape[2]
    scale = np.sqrt((A ** 2).sum(1, keepdims=True))
    scale[scale == 0] = 1
    Az = A / scale
    AtA = np.einsum('nki,nkj->nij', Az, Az) + 1e-9 * np.eye(k)[None]
    Atb = np.einsum('nki,nk->ni', Az, Y)
    beta = np.linalg.solve(AtA, Atb[..., None])[..., 0] / scale[:, 0, :]
    beta[count <= k] = np.nan
    res = [np.where(p['good'], p['y'] - (D[r] * beta[:, None, :]).sum(2), np.nan) for r, p in enumerate(parts)]
    rss = np.nansum(np.concatenate(res[:3], axis=1) ** 2, axis=1)
    rss[count <= k] = np.nan
    return beta, rss, res


def loss_integrals(p):
    """For each deactivation law, live catalyst E = c - beta * h(S); returns the integrals A = int h dS and
    B = int h/S dS from S to S0 that enter c*t (Duggleby 1986, eq. 5, and the same step for the other laws)."""
    s, s0, e, L = p['s'], p['s0'], p['e'], p['L']
    T = p['P0'] + s0
    fr = e - s * L
    lin = s0 * L - e
    cube = (s0 ** 2 * e - (s0 ** 3 - s ** 3) / 3) / 2
    square = (s0 ** 2 * L - (s0 ** 2 - s ** 2) / 2) / 2
    A = {'free': fr, 'sub': e ** 2 / 2, 'sub2': cube, 'pfree': T * fr - e ** 2 / 2, 'psub': T * e ** 2 / 2 - cube}
    B = {'free': L ** 2 / 2, 'sub': lin, 'sub2': square, 'pfree': T * L ** 2 / 2 - lin, 'psub': T * lin - square}
    return A, B


def inhibitor_integrals(p, kind, g):
    """Used-up inhibitor: E = c - I0 * h(S), with h = 1 - (S/S0)^g (binds free catalyst, M10) or
    h = 1 - exp(-g (S0 - S)) (binds the catalyst-substrate complex, M15)."""
    s, s0, e, L = p['s'], p['s0'], p['e'], p['L']
    if kind == 'free':
        return e - s0 / (g + 1) * (1 - (s / s0) ** (g + 1)), L - (1 - (s / s0) ** g) / g
    return e - (1 - np.exp(-g * e)) / g, L - np.exp(-g * s0) * (expi(g * s0) - expi(g * s))


def extra_term(A, B, a, b, c):
    """One extra column: the two integrals in the proportions of the plain fit (positive a and b only)."""
    return (a[:, None] * A + b[:, None] * B) / c


def best_index(rss):
    filled = np.where(np.isnan(rss), np.inf, rss)
    k = filled.argmin(1).astype(float)
    k[np.isinf(filled.min(1))] = np.nan
    return k


def _plain_fit(x1, runs):
    parts = law_parts(x1, runs)
    base, rss0, _ = fit_design(parts)
    a, b = np.maximum(np.nan_to_num(base[:, 0]), 0), np.maximum(np.nan_to_num(base[:, 1]), 0)
    n_good = sum(p['good'].sum(1) for p in parts[:3])
    ref = safe_div(sum(p['y'].sum(1) for p in parts[:3]), n_good.astype(float), n_good > 0)
    return parts, rss0, a, b, ref


def deactivation_laws(x1, runs, plain):
    """Catalyst deactivation laws (the "Death laws" family): the MM fit plus one catalyst-loss term per law.
    free: catalyst dies on its own (M9); sub: loss tracks substrate used (M11, M14); sub2: complex dies with a second
    substrate (M16); pfree, psub: deactivation with product (M12, M17); free_b, sub_b: pair deaths, fraction lost
    independent of loading (M13, M18, M19). law_rel = how much worse than the best law; law_loss = catalyst lost
    (relative to run 1's loading for the first five); law_res4 = mean residual in held-out run 4;
    law_best = index of the best law. Returns the columns and the best law's error (for inh_vs_law)."""
    parts, rss0, a, b, ref = plain
    ints = [loss_integrals(p) for p in parts]
    names, rss_all, fits = [], [], []
    for scaled, laws in ((True, LOSS_LAWS), (False, SHARE_LAWS)):
        for law in laws:
            z = [extra_term(i[0][law], i[1][law], a, b, p['c'] if scaled else 1.0) for i, p in zip(ints, parts)]
            beta, rss, res = fit_design(parts, [z])
            names.append(law if scaled else law + '_b')
            rss_all.append(rss)
            fits.append((beta[:, 2], res, scaled))
    rss_all = np.stack(rss_all, axis=1)
    best = np.nanmin(rss_all, axis=1)
    f = {}
    for j, (name, (gamma, res, scaled)) in enumerate(zip(names, fits)):
        f['law_rel_' + name] = safe_div(rss_all[:, j], best, best > 0) - 1
        f['law_loss_' + name] = np.clip(gamma / x1[:, 0] if scaled else gamma, -5, 5)
        f['law_res4_' + name] = nanmean_rows(res[3]) / ref
    f['law_best'] = best_index(rss_all)
    return f, best


def inhibitor_laws(x1, runs, plain, best_law_rss):
    """Inhibitor laws: a poison present from the start, (c - I) t = a(S0 - S) + b ln(S0/S), or an inhibitor that is
    used up (free or complex-binding form, shape g = 0.5, 2, 8). inh_gain = share of the plain fit's error removed;
    inh_amount = inhibitor amount relative to run 1's loading; inh_res4 = mean residual in run 4; inh_best = best
    shape; inh_vs_law = best inhibitor fit's error / best deactivation law's error."""
    parts, rss0, a, b, ref = plain
    n = len(x1)
    f = {}
    beta, rss_p, res = fit_design(parts, [[p['t'] for p in parts]])
    f['inh_gain_poison'] = 1 - safe_div(rss_p, rss0, rss0 > 0)
    f['inh_amount_poison'] = np.clip(beta[:, 2] / x1[:, 0], -5, 5)
    f['inh_res4_poison'] = nanmean_rows(res[3]) / ref
    all_inh = [rss_p]
    for kind in ('free', 'sub'):
        per_g, amounts, res4 = [], [], []
        for g in INHIBITOR_SHAPES:
            z = [extra_term(*inhibitor_integrals(p, kind, g), a, b, p['c']) for p in parts]
            beta, rss, res = fit_design(parts, [z])
            per_g.append(rss)
            amounts.append(beta[:, 2] / x1[:, 0])
            res4.append(nanmean_rows(res[3]) / ref)
            f['inh_gain_%s_%g' % (kind, g)] = 1 - safe_div(rss, rss0, rss0 > 0)
        per_g = np.stack(per_g, axis=1)
        k = best_index(per_g)
        pick = np.nan_to_num(k).astype(int)
        f['inh_best_' + kind] = k
        f['inh_amount_' + kind] = np.clip(np.stack(amounts, 1)[np.arange(n), pick], -5, 5)
        f['inh_res4_' + kind] = np.stack(res4, 1)[np.arange(n), pick]
        all_inh.append(np.nanmin(per_g, axis=1))
    f['inh_vs_law'] = np.clip(safe_div(np.nanmin(np.stack(all_inh, 1), axis=1), best_law_rss, best_law_rss > 0), 0, 20)
    return f


# ============================================================================================ mass balance
def tail_deficit(x1, runs):
    """Deficit D = 1 - S - P (material missing from solution, held on the catalyst), averaged over the last quarter
    of each run's points; shape (n, 4)."""
    T = runs[0][0].shape[1] - 1
    k = max(1, math.ceil(T / 4))
    return np.stack([(1 - S[:, -k:] - P[:, -k:]).mean(1) for (_, S, P) in runs], axis=1)


def mass_deficit(x1, runs):
    """Mass deficit: deficit_slope = slope of D against loading over runs 1-3, through the origin (material bound
    per unit of catalyst); deficit_per_cat_r = D / loading of each run."""
    D = tail_deficit(x1, runs)
    f = {'deficit_slope': (x1[:, :3] * D[:, :3]).sum(1) / (x1[:, :3] ** 2).sum(1)}
    for r in range(4):
        f['deficit_per_cat_%d' % (r + 1)] = D[:, r] / x1[:, r]
    return f


def deficit_km(x1, runs, shape_km, shape_sat):
    """Deficit Km fit: the bound substrate obeys D * S = eps * c * S - Km * D, solved over runs 1-3 (summed as it
    stands and multiplied by S, so noise averages out). dk_km1 assumes eps = 1; dk_km2 and dk_eps fit both;
    dk_rmse / dk_res_4 = misfit and run-4 residual; dk_km1_early / _late use S >= 0.5 / S < 0.5; dk_gap1/2 and
    dk_logratio compare with the curve-shape Km (mm_km); dk_sat = saturation at S = 0.5, dk_sat_gap = minus mm_sat."""
    parts = []
    for r in range(4):
        _, S, P = runs[r]
        S, P = S[:, 1:], P[:, 1:]
        good = S > 0.02
        S0 = np.where(good, S, 0.0)
        D = np.where(good, 1 - S - P, 0.0)
        parts.append((S0, D, np.where(good, x1[:, r:r + 1], 0.0), good))
    S, D, c, good = (np.concatenate([p[j] for p in parts[:3]], axis=1) for j in range(4))
    y, u, v = D * S, c * S, -D
    count = good.sum(1)
    ref = safe_div(u.sum(1), count.astype(float), count > 0)

    def one_km(w):
        below = (D * w).sum(1)
        return np.clip(safe_div((w * S * (c - D)).sum(1), below, np.abs(below) > 1e-9), -10, 50)

    km1 = one_km(1.0)
    m11, m12, r1 = u.sum(1), v.sum(1), y.sum(1)
    m21, m22, r2 = (u * S).sum(1), (v * S).sum(1), (y * S).sum(1)
    det = m11 * m22 - m12 * m21
    ok = (np.abs(det) > 1e-10 * np.abs(m11 * m22)) & (count >= 3)
    eps = safe_div(r1 * m22 - m12 * r2, det, ok)
    km2 = safe_div(m11 * r2 - m21 * r1, det, ok)
    res = y - eps[:, None] * u - km2[:, None] * v
    Sr, Dr, cr, g4 = parts[3]
    res4 = np.where(g4, Dr * Sr - eps[:, None] * cr * Sr + km2[:, None] * Dr, np.nan)
    f = {'dk_km1': km1, 'dk_km2': np.clip(km2, -10, 50), 'dk_eps': np.clip(eps, -2, 5),
         'dk_rmse': np.sqrt(safe_div(np.where(good, res ** 2, 0).sum(1), count - 2.0, count > 2)) / ref,
         'dk_res_4': nanmean_rows(res4) / ref,
         'dk_km1_early': one_km(S >= 0.5), 'dk_km1_late': one_km((S < 0.5) & good)}
    f['dk_gap1'] = np.clip(km1 - shape_km, -60, 60)
    f['dk_gap2'] = np.clip(f['dk_km2'] - shape_km, -60, 60)
    both = (km1 > 0) & (shape_km > 0)
    f['dk_logratio'] = np.log(safe_div(km1, shape_km, both))
    f['dk_sat'] = safe_div(np.full(len(km1), 0.5), km1 + 0.5, km1 > -0.45)
    f['dk_sat_gap'] = np.clip(f['dk_sat'] - shape_sat, -3, 3)
    return f


# ============================================================================================ context
def curve_shapes(x1, runs):
    """Shape of each run's progress curve (reaction progress kinetic analysis; Blackmond 2005), from product and,
    with the X_ prefix, from pooled conversion. P_end / t_end = last point; shape25 / shape50 = share of the run's
    gain reached by a quarter / half of its duration; t_half_frac = time to half the gain, as a share of the run."""
    f = {}
    for prefix, curve in (('', lambda run: run[2]), ('X_', pooled)):
        for r, run in enumerate(runs, start=1):
            t, y = run[0], curve(run)
            p0, p_end, t_end = y[:, 0], y[:, -1], t[:, -1]
            gain = p_end - p0
            f['%sP_end_%d' % (prefix, r)] = p_end
            if not prefix:
                f['t_end_%d' % r] = t_end
            for frac in (0.25, 0.5):
                mid = interp_rows(t, y, frac * t_end)
                f['%sshape%d_%d' % (prefix, 100 * frac, r)] = np.divide(mid - p0, gain, out=np.full(len(t), np.nan),
                                                                        where=gain > 0.05)
            t_half = cross_time(t, y, p0 + 0.5 * gain)
            f['%st_half_frac_%d' % (prefix, r)] = np.divide(t_half, t_end, out=np.full(len(t), np.nan),
                                                            where=t_end > 0)
    return f


def catalyst_loadings(x1, runs):
    """The loadings of runs 1-3 (inputs, not tests) and their spread, ln(cat0_3 / cat0_1)."""
    f = {'cat0_%d' % (r + 1): x1[:, r] for r in range(3)}
    f['cat_spread'] = np.log(x1[:, 2] / x1[:, 0])
    return f


def noise_and_sampling(x1, runs):
    """noise_est = measurement noise from the scatter of S + P within each run (S + P stays nearly constant);
    max_gap_r = longest gap between time points, as a share of the run's length."""
    T = runs[0][0].shape[1] - 1
    total = np.stack([S[:, 1:] + P[:, 1:] for (_, S, P) in runs], axis=2)
    res = total - total.mean(1, keepdims=True)
    f = {'noise_est': np.sqrt((res ** 2).sum((1, 2)) / (2 * 4 * (T - 1)))}
    for r, (t, _, _) in enumerate(runs, start=1):
        f['max_gap_%d' % r] = np.diff(t, axis=1).max(1) / np.maximum(t[:, -1], 1e-12)
    return f


# ============================================================================================ all features
def compute_features(x1, x2):
    """The 136 features of a batch of experiments that share their number of time points.
    x1: (n, 4) loadings; x2: (n, points, 12) sampled tables. Returns a DataFrame with the columns in COLUMNS."""
    x1 = np.asarray(x1, dtype=float)
    runs = sorted_runs(np.asarray(x2, dtype=float))
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        f = {}
        f.update(curve_shapes(x1, runs))
        f.update(catalyst_loadings(x1, runs))
        f.update(catalyst_order(x1, runs))
        f.update(same_excess(x1, runs))
        f.update(noise_and_sampling(x1, runs))
        f.update(mass_deficit(x1, runs))
        f.update(selwyn(x1, runs))
        f.update(vtna(x1, runs, f['noise_est']))
        f.update(mm_fit(x1, runs))
        plain = _plain_fit(x1, runs)
        laws, best_law_rss = deactivation_laws(x1, runs, plain)
        f.update(laws)
        f.update(inhibitor_laws(x1, runs, plain, best_law_rss))
        f.update(deficit_km(x1, runs, f['mm_km'], f['mm_sat']))
    return pd.DataFrame(f)[COLUMNS]


# ============================================================================================ families and feature sets
def families(cols):
    """{family: its columns}, in the order of config.FAMILY_RULES."""
    fam = {name: [] for name, _ in C.FAMILY_RULES}
    for c in cols:
        hit = [name for name, pre in C.FAMILY_RULES if c.startswith(pre)]
        assert hit, 'no family for ' + c
        fam[hit[0]].append(c)
    return fam


def keep_columns(feature_set='84'):
    """The model's columns: all 136, or the 84 left after the correlation filter (results/tables/drop_correlated.json,
    written by eda.py). Order is that of COLUMNS."""
    if feature_set == '136':
        return list(COLUMNS)
    with open(C.TABLES / 'drop_correlated.json') as f:
        drop = set(json.load(f)['dropped'])
    return [c for c in COLUMNS if c not in drop]


def clean(X):
    """Infinite values (a few ratios) are treated as missing."""
    X = np.array(X, dtype=np.float32)
    X[np.isinf(X)] = np.nan
    return X
