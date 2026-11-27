"""Step 6b (supporting, not in the report): example figures S1-S12, one per feature family, for the supporting
document (results/supporting/), drawn on real experiments so that each feature in its tables can point to the panel
that shows what it measures.

Examples are training experiments typical of their mechanism (closest to the mechanism's median on key features,
among 400 random experiments each), plus one noisy test experiment for the noise family. They are the full 20-point
runs from the dataset; the feature values printed on the panels are computed on exactly the plotted data by
features.compute_features, and fitted curves use the same fit functions. Needs data/raw.

Usage:  uv run python -m kinetic_mech.plots_appendix [1 5 12 ...]      (default: all) -> results/supporting/feature_examples/
"""
import re
import sys
import warnings

import numpy as np

from . import config as C
from . import features as ft
from . import style as ps
from .dataset import load_pickle, load_test, memmap_x2_train

ps.use()
import matplotlib.pyplot as plt                     # noqa: E402

warnings.filterwarnings('ignore')
OUT = C.SUPPORTING / 'feature_examples'
sorted_runs, cross_time = ft.sorted_runs, ft.cross_time
H = 2.2                                              # height of every appendix figure (in)
RUN_C = [ps.BLUE, ps.ORANGE, ps.GREEN, ps.YELLOW]
EX = {'M1': 3403804, 'M4': 4402961, 'M6': 1451153, 'M7': 3289583, 'M9': 4404298, 'M10': 1535737,
      'M12': 4760491, 'M14': 79226, 'M16': 1363188}
NOISY = (6, 5, 17)                                   # test set: 6 points, 5% noise, experiment index

_x1 = np.asarray(load_pickle('x1_train'))
_y = np.asarray(load_pickle('y_train')).ravel()
_x2 = memmap_x2_train(len(_y))


def sample(name):
    i = EX[name]
    assert 'M%d' % (_y[i] + 1) == name, (name, _y[i])
    x1, x2 = _x1[i:i + 1].astype(float), np.asarray(_x2[i:i + 1], float)
    return x1, x2, sorted_runs(x2), ft.compute_features(x1, x2).iloc[0]


def ext(run):
    t, S, P = run
    return (1 - S + P) / 2


def title(ax, letter, text):
    ax.set_title('(%s) %s' % (letter, text), fontsize=ps.PANEL_TITLE, loc='left')


def run_legend(fig, n=3, extra=()):
    h = [plt.Line2D([], [], color=RUN_C[r], marker='o' if r < 3 else 's', ms=3.5, lw=1, label='Run %d' % (r + 1))
         for r in range(n)] + list(extra)
    fig.legend(handles=h, loc='outside upper center', ncol=len(h), handlelength=1.8, columnspacing=1.2)


def num(v, d=2):
    """A number with a true minus sign."""
    return ('%.*f' % (d, v)).replace('-', '−')


def note(ax, text, loc=(0.97, 0.06), ha='right', va='bottom'):
    text = re.sub(r'(?<=[= ])-(?=\d)', '−', text)          # true minus signs in printed values
    ax.text(*loc, text, transform=ax.transAxes, ha=ha, va=va, fontsize=ps.ANNOT)


def save(fig, k):
    ps.save(fig, OUT, 'figS%d' % k)
    print('figure S%d done' % k, flush=True)


# ------------------------------------------------------------------------------------------- A1 VTNA
def a1():
    x1, x2, runs, F = sample('M4')
    n = F['vtna_order_all']
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, H), layout='constrained')
    for k, (ax, power, lab) in enumerate(zip(axes, [0, 1, n], ['$t$', '$[\\mathrm{cat}]_0\\,t$',
                                                             '$[\\mathrm{cat}]_0^{\\,n}\\,t$'])):
        for r in range(3):
            t = runs[r][0][0]
            ax.plot(t * x1[0, r] ** power, ext(runs[r])[0], '-o', ms=2.5, lw=0.9, color=RUN_C[r])
        ax.set_xlabel(lab)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3), useMathText=True)
        ax.set_xlim(left=0)
    title(axes[0], 'a', 'Raw time')
    title(axes[1], 'b', 'Order 1')
    title(axes[2], 'c', 'Best order, $n$ = %.2f' % n)
    axes[0].set_ylabel('Conversion, $X$')
    run_legend(fig)
    print('A1 M4: vtna_order_all %.3f, width %.3f, rms %.4f' % (n, F['vtna_width_all'], F['vtna_rms_all']))
    save(fig, 1)


# ------------------------------------------------------------------------------------------- A2 catalyst order
def a2():
    x1, x2, runs, F = sample('M4')
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    t02 = []
    for r in range(3):
        t, S, P = (a[0] for a in runs[r])
        ax.plot(t, P, '-o', ms=2.5, lw=0.9, color=RUN_C[r])
        tc = cross_time(t[None], P[None], np.array([P[0] + 0.2]))[0]
        t02.append(tc)
        ax.plot([tc, tc], [0, 0.2], ':', color=RUN_C[r], lw=0.9)
    ax.axhline(0.2, color=ps.MUTED, lw=0.6, ls='--')
    ax.set_xlim(0, max(t02) * 3)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Product, P')
    title(ax, 'a', 'Time to make 0.2 of product')
    ax = axes[1]
    lc, lt = np.log(x1[0, :3]), np.log(t02)
    ax.plot(lc, lt, 'o', ms=4, color=ps.INK, zorder=3)
    for r in range(3):
        ax.plot(lc[r], lt[r], 'o', ms=4.5, color=RUN_C[r], zorder=4)
    ax.plot(lc, lt, '-', color=ps.MUTED, lw=0.7, zorder=2)
    note(ax, '\n'.join('%s = %.2f' % (k, F[k]) for k in ('cat_order_12', 'cat_order_23', 'cat_order_13')),
         loc=(0.97, 0.95), va='top')
    ax.set_xlabel('ln [cat]$_0$')
    ax.set_ylabel('ln $t$ (P = 0.2)')
    title(ax, 'b', 'Order = $-$slope')
    run_legend(fig)
    print('A2 M4:', {k: round(F[k], 3) for k in ('cat_order_12', 'cat_order_23', 'cat_order_13')})
    save(fig, 2)


# ------------------------------------------------------------------------------------------- A3 same excess
def a3():
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, H), layout='constrained', sharey=True)
    for k, (ax, name, verdict) in enumerate(zip(axes, ['M1', 'M14', 'M6'],
                                                ['lines up', 'run 4 faster', 'run 4 slower'])):
        x1, x2, runs, F = sample(name)
        m = int(np.abs(x1[0, :3] - x1[0, 3]).argmin())
        t4, S4, P4 = (a[0] for a in runs[3])
        tm, Sm, Pm = (a[0] for a in runs[m])
        shift = cross_time(tm[None], Pm[None], np.array([P4[0]]))[0]
        keep = tm >= shift - 1e-12
        ax.plot(tm[keep] - shift, Pm[keep], '-o', ms=2.5, lw=0.9, color=ps.MUTED)
        ax.plot(t4, P4, '-s', ms=2.5, lw=0.9, color=RUN_C[3])
        ax.set_xlim(left=0)
        ax.set_xlabel('Time after shift')
        title(ax, 'abc'[k], '%s: %s' % (name, verdict))
        note(ax, 'same_excess_log_ratio = %.2f' % F['same_excess_log_ratio'])
        print('A3 %s: matching run %d, same_excess_log_ratio %.3f, mean %.3f' % (name, m + 1,
              F['same_excess_log_ratio'], F['same_excess_mean']))
    axes[0].set_ylabel('Product, P')
    h = [plt.Line2D([], [], color=ps.MUTED, marker='o', ms=3.5, lw=1, label='Matching run (1, 2 or 3), shifted'),
         plt.Line2D([], [], color=RUN_C[3], marker='s', ms=3.5, lw=1, label='Run 4 (same excess)')]
    fig.legend(handles=h, loc='outside upper center', ncol=2, handlelength=1.8)
    save(fig, 3)


# ------------------------------------------------------------------------------------------- A4 Selwyn
def a4():
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained', sharey=True)
    for k, (ax, name, verdict) in enumerate(zip(axes, ['M1', 'M6'], ['curves overlap', 'curves separate'])):
        x1, x2, runs, F = sample(name)
        for r in range(3):
            t = runs[r][0][0]
            ax.plot(x1[0, r] * t, ext(runs[r])[0], '-o', ms=2.5, lw=0.9, color=RUN_C[r])
        ax.set_xlim(left=0)
        ax.set_xlabel('$[\\mathrm{cat}]_0\\,t$')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3), useMathText=True)
        title(ax, 'ab'[k], '%s: %s' % (name, verdict))
        note(ax, 'selwyn_13 = %.3f\nselwyn_rms = %.3f' % (F['selwyn_13'], F['selwyn_rms']))
        print('A4 %s: selwyn_13 %.3f rms %.3f' % (name, F['selwyn_13'], F['selwyn_rms']))
    axes[0].set_ylabel('Conversion, $X$')
    run_legend(fig)
    save(fig, 4)


# ------------------------------------------------------------------------------------------- fitted laws
def fitted(x1, runs, kind='mm'):
    """Measured and fitted c*t at every usable point of runs 1-4 for the plain MM fit, the best deactivation law
    or the poison law; returns (parts, fitted values, the name of the extra term)."""
    parts = ft.law_parts(x1, runs)
    base, rss0, res0 = ft.fit_design(parts)
    if kind == 'mm':
        return parts, [p['y'] - r for p, r in zip(parts, res0)], 'Michaelis–Menten', None
    a, b = np.maximum(np.nan_to_num(base[:, 0]), 0), np.maximum(np.nan_to_num(base[:, 1]), 0)
    if kind == 'poison':
        beta, rss, res = ft.fit_design(parts, [[p['t'] for p in parts]])
        return parts, [p['y'] - r for p, r in zip(parts, res)], 'poison', None
    ints = [ft.loss_integrals(p) for p in parts]
    best, out = None, {}
    for scaled, laws in ((True, ft.LOSS_LAWS), (False, ft.SHARE_LAWS)):
        for law in laws:
            z = [ft.extra_term(i[0][law], i[1][law], a, b, p['c'] if scaled else 1.0) for i, p in zip(ints, parts)]
            beta, rss, res = ft.fit_design(parts, [z])
            name = law if scaled else law + '_b'
            out[name] = (rss[0], [p['y'] - r for p, r in zip(parts, res)])
    best = min(out, key=lambda k: out[k][0])
    return parts, out[best][1], best, {k: v[0] for k, v in out.items()}


def draw_fit(ax, parts, fit, runs_shown=(0, 1, 2, 3)):
    for r in runs_shown:
        p = parts[r]
        g = p['good'][0]
        X = 1 - p['s'][0][g]
        o = np.argsort(X)
        ax.plot(X, p['y'][0][g], 'o' if r < 3 else 's', ms=2.8, color=RUN_C[r], mfc=RUN_C[r] if r < 3 else 'white',
                mew=0.8, zorder=3)
        ax.plot(X[o], fit[r][0][g][o], '-' if r < 3 else '--', color=ps.INK, lw=0.8, zorder=2)
    ax.set_xlabel('Conversion, $X$')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def fit_legend(fig):
    h = [plt.Line2D([], [], color=RUN_C[r], marker='o', ls='none', ms=3.5, label='Run %d' % (r + 1)) for r in range(3)]
    h += [plt.Line2D([], [], color=RUN_C[3], marker='s', mfc='white', ls='none', ms=3.5, label='Run 4 (held out)'),
          plt.Line2D([], [], color=ps.INK, lw=0.8, label='Fit to runs 1–3'),
          plt.Line2D([], [], color=ps.INK, lw=0.8, ls='--', label='Prediction for run 4')]
    fig.legend(handles=h, loc='outside upper center', ncol=6, handlelength=1.6, columnspacing=0.9)


def a5():
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained')
    for k, (ax, name, verdict) in enumerate(zip(axes, ['M1', 'M9'], ['law fits', 'law misses'])):
        x1, x2, runs, F = sample(name)
        parts, fit, _, _ = fitted(x1, runs, 'mm')
        draw_fit(ax, parts, fit)
        title(ax, 'ab'[k], '%s: %s' % (name, verdict))
        note(ax, 'mm_rmse = %.3f\nmm_res_4 = %.3f' % (F['mm_rmse'], F['mm_res_4']), loc=(0.04, 0.95), ha='left',
             va='top')
        print('A5 %s: mm_km %.3f mm_rmse %.4f mm_res_4 %.4f' % (name, F['mm_km'], F['mm_rmse'], F['mm_res_4']))
    axes[0].set_ylabel('$[\\mathrm{cat}]_0\\,t$')
    fit_legend(fig)
    save(fig, 5)


def law_bars(ax, values, labels, xlabel, highlight):
    yy = np.arange(len(values))[::-1]
    ax.barh(yy, values, height=0.65, color=[ps.BLUE if l == highlight else ps.LIGHT_BLUE for l in labels])
    for yi, v in zip(yy, values):
        ax.text(v + 0.02 * max(values), yi, '%.2f' % v, va='center', fontsize=ps.ANNOT)
    ax.set_yticks(yy, labels)
    for t in ax.get_yticklabels():
        t.set_fontweight('bold' if t.get_text() == highlight else 'normal')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 1.18 * max(values))
    ax.set_xlabel(xlabel)


def a6():
    x1, x2, runs, F = sample('M12')
    parts, fit, best, rss = fitted(x1, runs, 'law')
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained', gridspec_kw={'width_ratios': [1.2, 1]})
    draw_fit(axes[0], parts, fit)
    axes[0].set_ylabel('$[\\mathrm{cat}]_0\\,t$')
    title(axes[0], 'a', 'M12: best deactivation law (%s)' % best)
    laws = list(ft.LOSS_LAWS) + [l + '_b' for l in ft.SHARE_LAWS]
    rel = [F['law_rel_' + l] for l in laws]
    law_bars(axes[1], rel, laws, 'law_rel (0 = best law)', best)
    title(axes[1], 'b', 'Fit of each law, relative to the best')
    fit_legend(fig)
    print('A6 M12: best %s, law_best %s, rel %s, loss %s' % (best, F['law_best'], np.round(rel, 3),
          {l: round(F['law_loss_' + l], 3) for l in laws}))
    save(fig, 6)


def a7():
    x1, x2, runs, F = sample('M10')
    parts, fit, _, _ = fitted(x1, runs, 'poison')
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained', gridspec_kw={'width_ratios': [1.2, 1]})
    draw_fit(axes[0], parts, fit)
    axes[0].set_ylabel('$[\\mathrm{cat}]_0\\,t$')
    title(axes[0], 'a', 'M10: poison fit (fixed catalyst loss)')
    note(axes[0], 'inh_amount_poison = %.2f' % F['inh_amount_poison'], loc=(0.04, 0.95), ha='left', va='top')
    keys = ['inh_gain_poison'] + ['inh_gain_%s_%g' % (k, g) for k in ('free', 'sub') for g in ft.INHIBITOR_SHAPES]
    labels = ['poison'] + ['%s, g = %g' % (k, g) for k in ('free', 'sub') for g in ft.INHIBITOR_SHAPES]
    gains = [F[k] for k in keys]
    law_bars(axes[1], gains, labels, 'Share of MM fit error removed', labels[int(np.argmax(gains))])
    title(axes[1], 'b', 'Gain of each inhibitor law')
    fit_legend(fig)
    print('A7 M10:', {k: round(F[k], 3) for k in keys + ['inh_amount_poison', 'inh_vs_law', 'inh_best_free']})
    save(fig, 7)


# ------------------------------------------------------------------------------------------- A8 mass deficit
def a8():
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained')
    x1, x2, runs, F = sample('M14')
    ax = axes[0]
    for r in range(4):
        t, S, P = (a[0] for a in runs[r])
        ax.plot(t / t[-1], 1 - S - P, '-o' if r < 3 else '-s', ms=2.5, lw=0.9, color=RUN_C[r])
    ax.axvspan(0.75, 1.0, color=ps.GRID, zorder=0, lw=0)
    ax.text(0.875, 0.97, 'averaged', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=ps.ANNOT)
    ax.set_xlabel('Time / run length')
    ax.set_ylabel('Deficit, $D = 1 - S - P$')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    title(ax, 'a', 'M14: substrate held on the catalyst')
    ax = axes[1]
    for k, name in enumerate(['M9', 'M14', 'M16']):
        x1, x2, runs, F = sample(name)
        T = runs[0][0].shape[1] - 1
        q = max(1, int(np.ceil(T / 4)))
        D = [(1 - runs[r][1][0, -q:] - runs[r][2][0, -q:]).mean() for r in range(3)]
        c = 100 * x1[0, :3]
        slope = F['deficit_slope']
        col, mk = [ps.INK, ps.MUTED, ps.RULE][k], 'os^'[k]
        cc = np.array([0, c.max() * 1.05])
        ax.plot(cc, slope * cc / 100, '-', lw=0.8, color=col)
        ax.plot(c, D, mk, ms=4, color=col, zorder=3, label='%s: deficit_slope = %.2f' % (name, slope))
        print('A8 %s: deficit_slope %.3f per_cat_1 %.3f' % (name, slope, F['deficit_per_cat_1']))
    ax.set_xlabel('Catalyst loading, [cat]$_0$ (mol%)')
    ax.set_ylabel('Deficit, last quarter')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', handlelength=1.0)
    title(ax, 'b', 'Deficit against loading, runs 1–3')
    run_legend(fig, 4)
    save(fig, 8)


# ------------------------------------------------------------------------------------------- A9 deficit Km
def a9():
    x1, x2, runs, F = sample('M1')
    eps, km = F['dk_eps'], F['dk_km2']
    fig, ax = plt.subplots(figsize=(ps.HALF + 0.6, H + 0.2), layout='constrained')
    for r in range(3):
        t, S, P = (a[0] for a in runs[r])
        g = S[1:] > 0.02
        ax.plot(S[1:][g], (1 - S - P)[1:][g] / x1[0, r], 'o', ms=3, color=RUN_C[r], label='Run %d' % (r + 1))
    s = np.linspace(0.02, 1, 100)
    ax.plot(s, eps * s / (km + s), '-', color=ps.INK, lw=0.9,
            label='$D/[\\mathrm{cat}]_0 = \\varepsilon S/(K_m + S)$')
    note(ax, 'dk_eps = %.2f\ndk_km2 = %.2f' % (eps, km))
    ax.set_xlabel('Substrate, S')
    ax.set_ylabel('Deficit / [cat]$_0$')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', handlelength=1.4)
    print('A9 M1: dk_eps %.3f dk_km2 %.3f dk_km1 %.3f mm_km %.3f' % (eps, km, F['dk_km1'], F['mm_km']))
    save(fig, 9)


# ------------------------------------------------------------------------------------------- A10 curve shapes
def a10():
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained')
    x1, x2, runs, F = sample('M9')
    ax = axes[0]
    t, S, P = (a[0] for a in runs[0])
    ax.plot(t, P, '-o', ms=2.5, lw=0.9, color=RUN_C[0])
    ax.plot([t[-1]], [P[-1]], 'o', ms=6, mfc='none', mec=ps.INK, mew=0.8)
    ax.annotate('P_end_1 = %.2f\nt_end_1 = %.2f' % (F['P_end_1'], F['t_end_1']), (t[-1], P[-1]), xytext=(-4, -24),
                textcoords='offset points', ha='right', va='top', fontsize=ps.ANNOT)
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Product, P')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    title(ax, 'a', 'M9, run 1: end point')
    ax = axes[1]
    for name, col in (('M9', RUN_C[0]), ('M7', ps.ORANGE)):
        x1, x2, runs, F = sample(name)
        t, S, P = (a[0] for a in runs[0])
        u, v = t / t[-1], (P - P[0]) / (P[-1] - P[0])
        ax.plot(u, v, '-o', ms=2.5, lw=0.9, color=col, label='%s: shape25 = %.2f, t_half_frac = %.2f'
                % (name, F['shape25_1'], F['t_half_frac_1']))
    for q in (0.25, 0.5):
        ax.axvline(q, color=ps.MUTED, lw=0.6, ls=':')
    ax.plot([0, 1], [0, 1], color=ps.GRID, lw=0.8, zorder=0)
    ax.axhline(0.5, color=ps.MUTED, lw=0.6, ls=':')
    ax.set_xlabel('Time / run length')
    ax.set_ylabel('Share of product gain')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', handlelength=1.4, frameon=True, framealpha=1, edgecolor='none', borderpad=0.3)
    title(ax, 'b', 'Fast start (M9) and slow start (M7)')
    save(fig, 10)


# ------------------------------------------------------------------------------------------- A11 loadings
def a11():
    rows = np.random.default_rng(1).choice(len(_x1), 200_000, replace=False)
    c = 100 * _x1[rows].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained')
    ax = axes[0]
    bins = np.logspace(np.log10(c[:, :3].min()), np.log10(c[:, :3].max()), 50)
    for r in range(3):
        ax.hist(c[:, r], bins=bins, histtype='step', lw=1.1, color=RUN_C[r], label='cat0_%d' % (r + 1))
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10], ['1', '2', '5', '10'])
    ax.minorticks_off()
    ax.set_xlabel('Catalyst loading (mol%)')
    ax.set_ylabel('Experiments')
    ax.legend(loc='upper left')
    title(ax, 'a', 'Loadings of runs 1–3')
    ax = axes[1]
    ax.hist(np.log(c[:, 2] / c[:, 0]), bins=50, color=ps.BLUE)
    ax.set_xlabel('cat_spread = ln(cat0_3 / cat0_1)')
    ax.set_ylabel('Experiments')
    title(ax, 'b', 'Spread of loadings')
    print('A11: loadings %.2f-%.2f mol%%, spread median %.2f' % (c[:, :3].min(), c[:, :3].max(),
          np.median(np.log(c[:, 2] / c[:, 0]))))
    save(fig, 11)


# ------------------------------------------------------------------------------------------- A12 noise and sampling
def a12():
    pts, noise, i = NOISY
    x1a, x2t, yt = load_test(pts, noise)
    x1, x2 = x1a[i:i + 1].astype(float), np.asarray(x2t[i:i + 1], float)
    runs = sorted_runs(x2)
    F = ft.compute_features(x1, x2).iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, H), layout='constrained')
    ax = axes[0]
    for r in range(4):
        t, S, P = (a[0] for a in runs[r])
        ax.plot(t / t[-1], S + P, '-o' if r < 3 else '-s', ms=3, lw=0.8, color=RUN_C[r])
    ax.set_xlabel('Time / run length')
    ax.set_ylabel('S + P')
    title(ax, 'a', 'Scatter of S + P (noise_est = %.3f)' % F['noise_est'])
    ax = axes[1]
    for r in range(4):
        t = runs[r][0][0] / runs[r][0][0][-1]
        ax.plot(t, np.full(len(t), r), 'o' if r < 3 else 's', ms=3.5, color=RUN_C[r])
        g = int(np.argmax(np.diff(t)))
        ax.plot(t[g:g + 2], [r, r], '-', color=RUN_C[r], lw=2.2, solid_capstyle='butt')
        ax.text(1.03, r, 'max_gap_%d = %.2f' % (r + 1, F['max_gap_%d' % (r + 1)]), va='center',
                fontsize=ps.ANNOT, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(4), ['Run %d' % (r + 1) for r in range(4)])
    ax.set_ylim(3.6, -0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time / run length')
    title(ax, 'b', 'Sampling times; thick line = longest gap')
    run_legend(fig, 4)
    print('A12: test %d pts %d%% #%d (M%d), noise_est %.4f' % (pts, noise, i, yt[i] + 1, F['noise_est']))
    save(fig, 12)


FIGS = {1: a1, 2: a2, 3: a3, 4: a4, 5: a5, 6: a6, 7: a7, 8: a8, 9: a9, 10: a10, 11: a11, 12: a12}

def main():
    for k in [int(a) for a in sys.argv[1:]] or list(FIGS):
        FIGS[k]()


if __name__ == '__main__':
    main()
