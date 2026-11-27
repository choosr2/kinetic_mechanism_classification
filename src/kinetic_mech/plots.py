"""Step 6: the figures, drawn from the saved tables in results/tables (no model is run).

In the report (results/report/, named as in the report):
  Figure_1_feature_families         Figure 1: the four runs of one training experiment and the 12 feature families
                                    computed from them (needs data/raw)
  Figure_2_models_and_permutation   Figure 2 (A) validation accuracy and log loss of every model, (B) test confusion
                                    matrix of the final LightGBM model, (C-E) permuting each feature family: (C)
                                    breadth and depth of the recall loss, (D) overall accuracy, (E) recall of each
                                    mechanism; 9.0 x 6.7 in
  Appendix_3_feature_distributions  Appendix 3: box plots of the 84 model features on a robust scale
  Appendix_4_pearson_correlation    Appendix 4: Pearson correlation of the 136 features, ordered by family

Supporting (results/supporting/figures and eda/; not in the report):
  pearson_136_square, example_run, models_validation, confusion_test_6pts_0pct, family_accuracy_loss,
  family_recall_change, family_breadth (the panels of Figure 2 drawn on their own), eda/class_balance,
  eda/data_quality

Differences in accuracy or recall are percentage points, written as %.
Usage:  uv run python -m kinetic_mech.plots [report | supporting | figure1 figure2 appendix3 appendix4 models ...]
        (default: all)
"""
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import style as st
from .features import families

st.use()
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.ticker import FixedLocator          # noqa: E402

OUT = C.SUPPORTING / 'figures'                     # supporting figures
OUT_EDA = C.SUPPORTING / 'eda'
PERM = C.TABLES / 'permutation'
EXAMPLE = 4210589                                   # an M11 training experiment with 20 points per run


def family_loss():
    """Accuracy loss (%) per family, mean of the three permutation repeats, sorted by loss."""
    runs = pd.read_csv(PERM / 'runs.csv')
    full = runs.loc[runs.unit == 'full', 'accuracy'].iloc[0]
    fam = runs[runs.unit.isin(C.FAMILY_ORDER)].groupby('unit').accuracy.agg(['mean', 'std'])
    fam['loss'] = 100 * (full - fam['mean'])
    return fam.sort_values('loss', ascending=False), 100 * full


# ------------------------------------------------------------------------- Pearson (Appendix 4; square: supporting)
def supporting_pearson_square():
    """The Pearson matrix as a square figure (the report uses the wide one, Appendix 4)."""
    R = pd.read_csv(C.TABLES / 'eda' / 'pearson_136.csv', index_col=0)
    fam = families(list(R.index))
    fam = {f: fam[f] for f in C.FAMILY_ORDER if fam[f]}
    order = [c for cs in fam.values() for c in cs]
    R = R.loc[order, order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(st.FULL, 5.4), layout='constrained')
    im = ax.imshow(R.to_numpy(), cmap=st.DIVERGING, vmin=-1, vmax=1, interpolation='nearest')
    edges = np.cumsum([len(cs) for cs in fam.values()])
    starts = np.r_[0, edges[:-1]]
    for e in edges[:-1]:
        ax.axhline(e - 0.5, color=st.INK, lw=0.4)
        ax.axvline(e - 0.5, color=st.INK, lw=0.4)
    ticks = (starts + edges - 1) / 2
    labels = ['%s (%d)' % (st.FAMILY_LABEL[f], len(cs)) for f, cs in fam.items()]
    ax.set_yticks(ticks, labels)
    ax.set_xticks(ticks, labels, rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    cb = fig.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], fraction=0.04, pad=0.02, aspect=30)
    cb.set_label('Pearson correlation coefficient, $r$')
    cb.outline.set_linewidth(0.4)
    st.save(fig, OUT, 'pearson_136_square')


def spread(pos, need, lo, hi):
    """Move sorted label positions apart until neighbours are at least `need` apart, each as little as possible,
    keeping them within [lo, hi]. Only labels move; the data do not."""
    p = np.array(pos, float)
    for _ in range(200):
        for i in range(len(p) - 1):
            gap = p[i + 1] - p[i]
            if gap < need:
                p[i] -= (need - gap) / 2
                p[i + 1] += (need - gap) / 2
        p = np.clip(p, lo, hi)
    return p


def appendix4_pearson_correlation():
    """Report Appendix 4: Pearson correlation of the 136 features, ordered by family; 9.0 x 4.3 in (landscape).
    The cells are wider than tall, the family names along the bottom are at 45 degrees and all text is 10 pt. Where
    two small families sit next to each other, their names on the left are moved apart slightly (at most 0.02 in)."""
    LABEL = 10
    R = pd.read_csv(C.TABLES / 'eda' / 'pearson_136.csv', index_col=0)
    fam = families(list(R.index))
    fam = {f: fam[f] for f in C.FAMILY_ORDER if fam[f]}
    order = [c for cs in fam.values() for c in cs]
    R = R.loc[order, order]
    n = len(order)
    W, H = 9.0, 4.3
    left, right, bottom, top = 1.5, 8.05, 1.02, H - 0.06
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, (right - left) / W, (top - bottom) / H])
    im = ax.imshow(R.to_numpy(), cmap=st.DIVERGING, vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
    edges = np.cumsum([len(cs) for cs in fam.values()])
    starts = np.r_[0, edges[:-1]]
    for e in edges[:-1]:
        ax.axhline(e - 0.5, color=st.INK, lw=0.4)
        ax.axvline(e - 0.5, color=st.INK, lw=0.4)
    ticks = (starts + edges - 1) / 2
    row_in = (top - bottom) / n                                         # height of one feature row (in)
    y_ticks = spread(ticks, 0.95 * LABEL / 72 / row_in, -0.5, n - 0.5)  # left names at least one text line apart
    ax.set_yticks(y_ticks, ['%s (%d)' % (st.FAMILY_LABEL[f], len(cs)) for f, cs in fam.items()])
    ax.set_xticks(ticks, [st.FAMILY_LABEL[f] for f in fam], rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(length=0, labelsize=LABEL)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    cax = fig.add_axes([(right + 0.12) / W, bottom / H, 0.08 / W, (top - bottom) / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=LABEL)
    cb.set_label('Pearson correlation coefficient, $r$', fontsize=LABEL)
    cb.outline.set_linewidth(0.4)
    st.save(fig, C.REPORT, 'Appendix_4_pearson_correlation')


# ----------------------------------------------------------------------------- distributions (Appendix 3)
def appendix3_feature_distributions():
    """Report Appendix 3. Horizontal box-and-whisker plot of the 84 features, one labelled row each, in two columns of families.
    Box = Q1-Q3 with the median (white), whiskers = 1st-99th percentile, on (value - median) / IQR."""
    D = pd.read_csv(C.TABLES / 'eda' / 'distribution_84.csv')
    D['rank'] = D.family.map({f: i for i, f in enumerate(C.FAMILY_ORDER)})
    D = D.sort_values('rank', kind='stable').reset_index(drop=True)
    for k in ('q1', 'q3', 'p1', 'p99'):
        D['z_' + k] = (D[k] - D['median']) / D['scale']
    split = C.FAMILY_ORDER.index('Death laws') + 1             # graphical tests and fitted laws | the rest
    halves = [D[D['rank'] < split], D[D['rank'] >= split]]
    n_rows = max(len(h) + h['rank'].nunique() for h in halves)
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL, 6.7), layout='constrained', sharex=True)
    for ax, H in zip(axes, halves):
        ticks, labels, heads, ys, row = [], [], [], [], 0
        for f, g in H.groupby('rank', sort=True):
            ticks.append(row)
            labels.append(st.bold_math('%s (%d)' % (st.FAMILY_LABEL[C.FAMILY_ORDER[f]], len(g))))
            heads.append(len(ticks) - 1)
            row += 1
            for feat in g.feature:
                ticks.append(row)
                labels.append(feat)
                ys.append(row)
                row += 1
        y = np.array(ys)
        ax.hlines(y, H.z_p1, H.z_p99, color=st.MUTED, lw=0.6, zorder=2)
        for k in ('z_p1', 'z_p99'):
            ax.vlines(H[k], y - 0.25, y + 0.25, color=st.MUTED, lw=0.6, zorder=2)
        ax.barh(y, H.z_q3 - H.z_q1, left=H.z_q1, height=0.62, color=st.BLUE, edgecolor='none', zorder=3)
        ax.vlines(np.zeros(len(y)), y - 0.31, y + 0.31, color='white', lw=0.8, zorder=4)
        ax.axvline(0, color=st.GRID, lw=0.6, zorder=1)
        ax.set_yticks(ticks, labels)
        for k in heads:
            ax.get_yticklabels()[k].set_fontweight('bold')
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(n_rows - 0.5, -0.7)
        ax.spines['left'].set_visible(False)
        ax.set_xscale('symlog', linthresh=1, linscale=0.6)
        ax.set_xlim(-150, 150)
        ax.xaxis.set_major_locator(FixedLocator([-100, -10, -1, 0, 1, 10, 100]))
        ax.set_xticklabels(['$-10^2$', '$-10$', '$-1$', '0', '1', '10', '$10^2$'])
        ax.xaxis.set_minor_locator(FixedLocator([]))
    fig.supxlabel('(Value $-$ median) / IQR', fontsize=st.AXIS_LABEL)
    st.save(fig, C.REPORT, 'Appendix_3_feature_distributions')


# ------------------------------------------------------------------------------- example run (supporting)
def supporting_example_run():
    from .dataset import load_train
    x1, x2, y = load_train()
    data, cat = np.asarray(x2[EXAMPLE]), x1[EXAMPLE]
    fig, axes = plt.subplots(1, 4, figsize=(st.FULL, 2.25), sharey=True, layout='constrained')
    titles = ['Run 1', 'Run 2', 'Run 3', 'Run 4 (same excess)']
    for r, ax in enumerate(axes):
        t, s, p = data[:, 3 * r], data[:, 3 * r + 1], data[:, 3 * r + 2]
        o = np.argsort(t, kind='stable')
        ax.plot(t[o], s[o], '-o', ms=3, color=st.BLUE, lw=1, label='Substrate, S')
        ax.plot(t[o], p[o], '--o', ms=3, mfc='white', color=st.ORANGE, lw=1, label='Product, P')
        ax.set_title('%s\n[cat]$_0$ = %.2f mol%%' % (titles[r], 100 * cat[r]), fontsize=st.PANEL_TITLE)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_ylabel('Normalised concentration')
    fig.supxlabel('Normalised time', fontsize=st.AXIS_LABEL)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, handlelength=2.2)
    print('example run: training experiment %d, %s' % (EXAMPLE, C.MECHS[int(y[EXAMPLE])]))
    st.save(fig, OUT, 'example_run')


# ------------------------------------------------------------------------ runs -> feature families (Figure 1)
FEATURE_TABLE = [   # family, what is computed, source (reference number in the paper)
    ('VTNA', 'Catalyst order from overlay of runs 1–3; overlay error, residuals', '3'),
    ('Catalyst order (pairs)', 'Order from times to 20% conversion, between pairs of kinetic runs', '1'),
    ('Same-excess', 'Mismatch of run 4 with the time-shifted run at the same loading', '8'),
    ('Selwyn', 'Gaps between runs 1–3 plotted against [cat]$_0t$', '15'),
    ('Michaelis–Menten (MM) fit', 'Integrated MM fit to runs 1–3: $K_\\mathrm{m}$, misfit, residuals incl. run 4',
     '16'),
    ('Deactivation laws', 'MM fit plus one catalyst-loss term per law: misfit, loss, run-4 residual', '17'),
    ('Inhibitor laws', 'MM fit with an initial poison: amount, fit gain, run-4 residual', '17'),
    ('Mass deficit', 'Late $S_0 - S - P$ per unit catalyst; slope against loading', '–'),
    ('Deficit $K_\\mathrm{m}$ fit', 'Binding fit of the deficit against [$S$]: binding constants, saturation', '18'),
    ('Curve shapes', 'Final conversion, duration and early product fraction of each run', '8'),
    ('Catalyst loadings', 'Loadings of runs 1–3 and their log-ratio', '–'),
    ('Noise and sampling', 'Noise estimate from $S + P$ scatter; largest sampling gap', '–'),
]


def wrap_text(fig, text, width_in, size, weight='normal'):
    """Split text into lines no wider than width_in (measured with the renderer); $...$ math stays whole."""
    import re
    renderer = fig.canvas.get_renderer()
    words = re.findall(r'\$[^$]*\$\S*|\S+', text)

    def width(s):
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return w

    lines, cur = [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if cur and width(trial) > width_in:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur]


def figure1_feature_families():
    """Report Figure 1: the four runs of one training experiment, stacked, and an arrow to the table of
    feature families computed from them (needs data/raw). Drawn on an inch grid, so every text region keeps the
    paper's sizes."""
    from matplotlib.patches import FancyArrowPatch
    from .dataset import load_train
    x1, x2, y = load_train()
    data, cat = np.asarray(x2[EXAMPLE]), x1[EXAMPLE]
    # vertical layout of the runs column (inches): legend, then 4 panels, each with a title above it
    ph, title, ticks = 0.36, 0.15, 0.2              # panel height, one title line, x tick labels
    titles = ['Run 1, [cat]$_0$ = %.2f mol%%' % (100 * cat[0]), 'Run 2, [cat]$_0$ = %.2f mol%%' % (100 * cat[1]),
              'Run 3, [cat]$_0$ = %.2f mol%%' % (100 * cat[2]),
              'Run 4 (same excess),\n[cat]$_0$ = %.2f mol%%' % (100 * cat[3])]
    n_lines = [t.count('\n') + 1 for t in titles]
    head = 0.27                                     # room for the legend
    H = head + sum(n * title + ph + ticks for n in n_lines) + 0.22
    W = st.FULL
    fig = plt.figure(figsize=(W, H))
    box = lambda l, b, w, h: [l / W, b / H, w / W, h / H]                   # noqa: E731  inches -> figure
    # --- stacked runs
    left, width = 0.55, 1.0
    axes, yy = [], H - head
    for r in range(4):
        yy -= n_lines[r] * title + ph
        ax = fig.add_axes(box(left, yy, width, ph))
        yy -= ticks
        t, s, p = data[:, 3 * r], data[:, 3 * r + 1], data[:, 3 * r + 2]
        o = np.argsort(t, kind='stable')
        ax.plot(t[o], s[o], '-o', ms=2.2, color=st.BLUE, lw=0.9, label='Substrate, S')
        ax.plot(t[o], p[o], '--o', ms=2.2, mfc='white', mew=0.7, color=st.ORANGE, lw=0.9, label='Product, P')
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.text(0, 1.04, titles[r], transform=ax.transAxes, ha='left', va='bottom', fontsize=st.ANNOT,
                linespacing=1.1)
        axes.append(ax)
    bottom = axes[-1].get_position().y0 * H
    top = axes[0].get_position().y1 * H
    fig.text(0.1 / W, (top + bottom) / 2 / H, 'Normalised concentration', rotation=90, ha='left', va='center',
             fontsize=st.AXIS_LABEL)
    fig.text((left + width / 2) / W, 0.03 / H, 'Normalised time', ha='center', va='bottom', fontsize=st.AXIS_LABEL)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', ncol=2, handlelength=1.8, columnspacing=0.8,
               bbox_to_anchor=(0.02 / W, 1.0), borderaxespad=0.1, labelspacing=0.2)
    # --- arrow, level with run 2
    p2 = axes[1].get_position()
    a0, a1, ay = left + width + 0.08, left + width + 0.53, (p2.y0 + p2.y1) / 2 * H
    fig.add_artist(FancyArrowPatch((a0 / W, ay / H), (a1 / W, ay / H), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=11, lw=1.1, color=st.INK))
    fig.text((a0 + a1) / 2 / W, (ay - 0.07) / H, 'Feature\nengineering', ha='center', va='top',
             fontsize=st.ANNOT, linespacing=1.1)
    # --- table of feature families, spread to the height of the runs column
    tx = a1 + 0.12
    c_fam, c_what, c_src = tx, tx + 1.02, W - 0.04                      # column x (source right-aligned)
    w_fam, w_what = 0.98, c_src - 0.13 - (tx + 1.02)
    line = st.TICK * 1.22 / 72                                          # line height (in)
    rows = [(wrap_text(fig, f, w_fam, st.TICK), wrap_text(fig, w, w_what, st.TICK), s) for f, w, s in FEATURE_TABLE]
    y0, y_end = H - 0.08, bottom - 0.02
    used = 0.06 + line + 0.06 + 0.05 + sum(max(len(a), len(b)) * line for a, b, _ in rows)
    pad = min(0.14, (y0 - y_end - used) / len(rows))                    # space between rows
    fig.add_artist(plt.Line2D([tx / W, c_src / W], [y0 / H, y0 / H], lw=0.8, color=st.INK))
    yy = y0 - 0.06
    for x, txt, ha in ((c_fam, 'Family', 'left'), (c_what, 'What is computed', 'left'), (c_src, 'Source', 'right')):
        fig.text(x / W, yy / H, txt, ha=ha, va='top', fontsize=st.TICK, fontweight='bold')
    yy -= line + 0.06
    fig.add_artist(plt.Line2D([tx / W, c_src / W], [yy / H, yy / H], lw=0.5, color=st.INK))
    yy -= 0.05 + pad / 2
    for lf, lw_, src in rows:
        for k, ln in enumerate(lf):
            fig.text(c_fam / W, (yy - k * line) / H, ln, va='top', fontsize=st.TICK)
        for k, ln in enumerate(lw_):
            fig.text(c_what / W, (yy - k * line) / H, ln, va='top', fontsize=st.TICK)
        fig.text(c_src / W, yy / H, src, ha='right', va='top', fontsize=st.TICK)
        yy -= max(len(lf), len(lw_)) * line + pad
    yy += pad / 2
    fig.add_artist(plt.Line2D([tx / W, c_src / W], [yy / H, yy / H], lw=0.8, color=st.INK))
    print('Figure 1: training experiment %d, %s; %.2f x %.2f in' % (EXAMPLE, C.MECHS[int(y[EXAMPLE])], W, H))
    st.save(fig, C.REPORT, 'Figure_1_feature_families')


# ------------------------------------------------------------- Figure 2 panels on their own (supporting)
MODEL_ROWS = [('Dummy (class prior)', 'Dummy (class prior)'),
              ('Decision tree (204,800 rows)', 'Decision tree'),
              ('Random forest (204,800 rows)', 'Random forest'),
              ('Gradient boosting (scikit-learn) (204,800 rows)', 'HGB (scikit-learn)'),
              ('XGBoost (204,800 rows)', 'XGBoost'),
              ('LightGBM (204,800 rows)', 'LightGBM'),
              ('LightGBM, final (4,947,968 rows)', 'LightGBM, final')]


def supporting_models():
    """Validation accuracy and validation log loss of every model, side by side (two measures, two axes)."""
    res = pd.read_csv(C.TABLES / 'results.csv')
    val = res[res.set == 'validation'].set_index('model')
    acc = [100 * val.loc[m, 'accuracy'] for m, _ in MODEL_ROWS]
    ll = [val.loc[m, 'logloss_bits'] for m, _ in MODEL_ROWS]
    final_test = 100 * res[(res.model == MODEL_ROWS[-1][0]) & (res.set == C.MAIN_TEST)].accuracy.iloc[0]
    names = [n for _, n in MODEL_ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL, 2.4), layout='constrained', sharey=True)
    yy = np.arange(len(names))[::-1]
    for ax, v, fmt, top, lab, title in ((axes[0], acc, '%.1f', 100, 'Accuracy (%)', '(a) Accuracy'),
                                        (axes[1], ll, '%.2f', 7.5, 'Log loss (bits, lower is better)',
                                         '(b) Log loss')):
        ax.barh(yy, v, height=0.62, color=st.BLUE)
        for k, (yi, x) in enumerate(zip(yy, v)):
            text = fmt % x
            if ax is axes[0] and k == len(v) - 1:               # the final model: add its test accuracy
                text += ' (test %.1f)' % final_test
            ax.text(x + top * 0.012, yi, text, va='center', fontsize=st.ANNOT)
        ax.set_xlim(0, top)
        ax.set_xlabel(lab)
        ax.set_title(title, loc='left', fontsize=st.PANEL_TITLE)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(yy, names)
    st.save(fig, OUT, 'models_validation')


# ----------------------------------------------------------------------------------------------- 5. confusion
def supporting_confusion():
    cm = pd.read_csv(C.TABLES / ('confusion_%s.csv' % C.MAIN_TEST), index_col=0).to_numpy(float)
    pct = 100 * cm / cm.sum(1, keepdims=True)
    n = 20
    fig, ax = plt.subplots(figsize=(4.0, 3.75), layout='constrained')
    im = ax.imshow(pct, cmap=st.SEQUENTIAL, vmin=0, vmax=100)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='#c8c6c0', lw=0.4)         # light grey cell lines, to follow rows and columns
    ax.set_axisbelow(False)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(n), C.MECHS, rotation=90)
    ax.set_yticks(range(n), C.MECHS)
    for i in range(n):
        for j in range(n):
            if pct[i, j] >= 1:
                ax.text(j, i, '%d' % round(pct[i, j]), ha='center', va='center', fontsize=st.CELL,
                        color='white' if pct[i, j] >= 50 else st.INK)
    ax.set_xlabel('Predicted mechanism')
    ax.set_ylabel('Actual mechanism')
    cb = fig.colorbar(im, ax=ax, ticks=range(0, 101, 20), fraction=0.04, pad=0.02, aspect=25, shrink=0.84)
    cb.set_label('Share of experiments (%)')
    cb.outline.set_linewidth(0.4)
    st.save(fig, OUT, 'confusion_test_6pts_0pct')


# ----------------------------------------------------------------------------------------------- 6. family accuracy
def supporting_family_accuracy():
    fam, _ = family_loss()
    fig, ax = plt.subplots(figsize=(st.HALF, 2.75), layout='constrained')
    yy = np.arange(len(fam))[::-1]
    ax.barh(yy, fam.loss, height=0.68, color=st.BLUE)
    for yi, v in zip(yy, fam.loss):
        ax.text(v + 0.4, yi, '%.1f' % v, va='center', fontsize=st.ANNOT)
    ax.set_yticks(yy, [st.FAMILY_LABEL[f] for f in fam.index])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 28)
    ax.set_xlabel('Accuracy loss (%)')
    st.save(fig, OUT, 'family_accuracy_loss')


# ----------------------------------------------------------------------------------------------- 7. family x mechanism
def supporting_family_recall():
    fam, _ = family_loss()
    rc = pd.read_csv(PERM / 'recall_change.csv')
    M = rc.pivot(index='unit', columns='mech', values='change_points').loc[fam.index, C.MECHS]
    v = M.to_numpy()
    fig = plt.figure(figsize=(st.FULL, 3.0))
    ax = fig.add_axes([0.165, 0.09, 0.73, 0.75])
    cax = fig.add_axes([0.915, 0.09, 0.014, 0.75])
    im = ax.imshow(v, cmap=st.DIVERGING, vmin=-70, vmax=70, aspect='auto')
    nr, nc = v.shape
    ax.set_xticks(np.arange(nc + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nr + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(nc), C.MECHS)
    ax.set_yticks(range(nr), [st.FAMILY_LABEL[f] for f in M.index])
    for _, a, _b in C.CATEGORIES[1:]:
        ax.axvline(a - 0.5, color=st.INK, lw=0.6)
    for i in range(nr):
        for j in range(nc):
            if abs(v[i, j]) >= 5:
                ax.text(j, i, ('+%d' if v[i, j] > 0 else '−%d') % round(abs(v[i, j])),
                        ha='center', va='center', fontsize=st.CELL,
                        color='white' if abs(v[i, j]) >= 40 else st.INK)
    st.category_brackets(ax, where=1.02)
    cb = fig.colorbar(im, cax=cax, ticks=[-60, -40, -20, 0, 20, 40, 60])
    cb.set_label('Change in recall (%)')
    cb.outline.set_linewidth(0.4)
    st.save(fig, OUT, 'family_recall_change')


# ----------------------------------------------------------------------------------------------- 8. breadth vs peak
LABEL_AT = {  # family: label position in data units (x, y) and alignment; a thin line joins it to its point
    'Mass deficit': (14.0, 76, 'right'), 'Curve shapes': (20.5, 70, 'right'),
    'Death laws': (20.5, 54, 'right'), 'Deficit Km fit': (20.5, 42, 'right'),
    'Integrated fit (MM)': (0.2, 70, 'left'), 'Pooled order (VTNA)': (8.0, 63, 'left'),
    'Same-excess test': (0.2, 52, 'left'), 'Selwyn test': (3.5, 23, 'left'), 'Inhibitor laws': (9.5, 32, 'left'),
    'Noise and sampling': (0.2, 16, 'left'), 'Catalyst order (run pairs)': (3.0, 9, 'left'),
    'Catalyst loadings': (3.0, 1.5, 'left')}


def supporting_family_breadth():
    """Points at their exact values; hollow markers, so two points at nearly the same place both stay visible."""
    S = pd.read_csv(PERM / 'family_summary.csv')
    S = S[S.kind == 'family'].set_index('unit')
    fig, ax = plt.subplots(figsize=(st.HALF, 2.75), layout='constrained')
    ax.scatter(S.mechanisms_affected, S.largest_loss_points, s=20, facecolor='none', edgecolor=st.BLUE,
               linewidth=1.0, zorder=3)
    for f, r in S.iterrows():
        tx, ty, ha = LABEL_AT[f]
        lab = st.FAMILY_LABEL[f] + (' (%s)' % r.largest_loss_mech if isinstance(r.largest_loss_mech, str) else '')
        ax.annotate(lab, (r.mechanisms_affected, r.largest_loss_points), xytext=(tx, ty), textcoords='data', ha=ha,
                    va='center', fontsize=st.ANNOT, zorder=4,
                    arrowprops=dict(arrowstyle='-', color=st.RULE, lw=0.5, shrinkA=1.5, shrinkB=3))
    ax.set_xlim(-0.8, 20.5)
    ax.set_ylim(-3, 80)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel('Mechanisms losing $\\geq$5% recall')
    ax.set_ylabel('Largest recall loss on one\nmechanism (%)')
    st.save(fig, OUT, 'family_breadth')


# ----------------------------------------------------------------------------------------- Figure 2
COLUMN_LABEL = {  # family names on two lines, for the narrow bars of panel D
    'Mass deficit': 'Mass\ndeficit', 'Death laws': 'Deactivation\nlaws', 'Deficit Km fit': 'Deficit\n$K_\\mathrm{m}$ fit',
    'Curve shapes': 'Curve\nshapes', 'Inhibitor laws': 'Inhibitor\nlaws', 'Pooled order (VTNA)': 'VTNA',
    'Integrated fit (MM)': 'MM fit', 'Same-excess test': 'Same-\nexcess test', 'Selwyn test': 'Selwyn\ntest',
    'Noise and sampling': 'Noise and\nsampling', 'Catalyst order (run pairs)': 'Catalyst\norder',
    'Catalyst loadings': 'Catalyst\nloadings'}


def figure2_models_and_permutation():
    """Report Figure 2: the five panels in one landscape figure (9.0 x 6.7 in, for a landscape page at 100%), lettered in reading order:
    A models (fig 4) | B confusion matrix (fig 5)
    C breadth of each family's effect (fig 8) over D accuracy loss per family (fig 6) | E change in recall per family
    and mechanism (fig 7). Drawn on an inch grid so the text sizes match the other figures."""
    W, H = 9.0, 6.7
    fig = plt.figure(figsize=(W, H))
    box = lambda l, b, w, h: fig.add_axes([l / W, b / H, w / W, h / H])     # noqa: E731  inches -> axes

    def letter(x, y, s):
        fig.text(x / W, y / H, s, ha='left', va='top', fontsize=st.PANEL_LETTER, fontweight='bold')

    row = 3.5                                       # top of the bottom row
    fam, _ = family_loss()
    n_f = len(fam)

    # ---------------- A: validation accuracy and log loss of every model, side by side (fig 4)
    res = pd.read_csv(C.TABLES / 'results.csv')
    val = res[res.set == 'validation'].set_index('model')
    acc = [100 * val.loc[m, 'accuracy'] for m, _ in MODEL_ROWS]
    ll = [val.loc[m, 'logloss_bits'] for m, _ in MODEL_ROWS]
    final_test = 100 * res[(res.model == MODEL_ROWS[-1][0]) & (res.set == C.MAIN_TEST)].accuracy.iloc[0]
    names = [n for _, n in MODEL_ROWS]
    yy = np.arange(len(names))[::-1]
    for k_ax, (ax, v, fmt, top, lab) in enumerate(((box(1.1, 4.05, 1.8, 2.3), acc, '%.1f', 100,
                                                     'Validation accuracy (%)'),
                                                    (box(3.5, 4.05, 1.35, 2.3), ll, '%.2f', 7.5,
                                                     'Validation log loss (bits)'))):
        ax.barh(yy, v, height=0.45, color=st.BLUE)
        for k, (yi, x) in enumerate(zip(yy, v)):
            text = fmt % x + (' (test %.1f)' % final_test if top == 100 and k == len(v) - 1 else '')
            ax.text(x + top * 0.015, yi, text, va='center', fontsize=st.ANNOT, clip_on=False)
        ax.set_xlim(0, top)
        ax.set_xlabel(lab, labelpad=2)
        ax.set_yticks(yy, names if k_ax == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    letter(0.0, H, 'A')

    # ---------------- B: confusion matrix of the final model on the main test set (fig 5)
    cell = 0.13
    m_left, m_bot = 5.65, 4.0
    cm = pd.read_csv(C.TABLES / ('confusion_%s.csv' % C.MAIN_TEST), index_col=0).to_numpy(float)
    pct = 100 * cm / cm.sum(1, keepdims=True)
    ax = box(m_left, m_bot, 20 * cell, 20 * cell)
    im = ax.imshow(pct, cmap=st.SEQUENTIAL, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(np.arange(21) - 0.5, minor=True)
    ax.set_yticks(np.arange(21) - 0.5, minor=True)
    ax.grid(which='minor', color='#c8c6c0', lw=0.4)
    ax.set_axisbelow(False)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(20), C.MECHS, rotation=90)
    ax.set_yticks(range(20), C.MECHS)
    for i in range(20):
        for j in range(20):
            if pct[i, j] >= 1:
                ax.text(j, i, '%d' % round(pct[i, j]), ha='center', va='center', fontsize=st.CELL,
                        color='white' if pct[i, j] >= 50 else st.INK)
    ax.set_xlabel('Predicted mechanism', labelpad=2)
    ax.set_ylabel('Actual mechanism', labelpad=2)
    cax = box(m_left + 20 * cell + 0.1, m_bot, 0.06, 20 * cell)
    cb = fig.colorbar(im, cax=cax, ticks=range(0, 101, 20))
    cb.set_label('Share of experiments (%)', labelpad=2)
    cb.outline.set_linewidth(0.4)
    letter(5.15, H, 'B')

    # ---------------- C: largest recall loss against number of mechanisms affected (fig 8)
    S = pd.read_csv(PERM / 'family_summary.csv')
    S = S[S.kind == 'family'].set_index('unit')
    ax = box(0.95, 1.86, 2.5, 1.5)
    at = dict(LABEL_AT, **{'Mass deficit': (11.5, 76, 'right'), 'Curve shapes': (20.5, 72.5, 'right')})
    ax.scatter(S.mechanisms_affected, S.largest_loss_points, s=20, facecolor='none', edgecolor=st.BLUE,
               linewidth=1.0, zorder=3)
    for f, r in S.iterrows():
        tx, ty, ha = at[f]
        lab = st.FAMILY_LABEL[f] + (' (%s)' % r.largest_loss_mech if isinstance(r.largest_loss_mech, str) else '')
        ax.annotate(lab, (r.mechanisms_affected, r.largest_loss_points), xytext=(tx, ty), textcoords='data', ha=ha,
                    va='center', fontsize=st.ANNOT, zorder=4,
                    arrowprops=dict(arrowstyle='-', color=st.RULE, lw=0.5, shrinkA=1.5, shrinkB=3))
    ax.set_xlim(-0.8, 20.5)
    ax.set_ylim(-3, 80)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel('Mechanisms losing $\\geq$5% recall', labelpad=2)
    ax.set_ylabel('Largest recall loss on one\nmechanism (%)')
    letter(0.0, row, 'C')

    # ---------------- D: accuracy loss when each family is permuted (fig 6), under C
    ax = box(0.6, 0.75, 2.9, 0.33)
    ax.bar(range(n_f), fam.loss, width=0.7, color=st.BLUE)
    for k, v in enumerate(fam.loss):
        ax.text(k, v + 1.0, '%.1f' % v, ha='center', va='bottom', fontsize=st.ANNOT, clip_on=False)
    ax.set_xlim(-0.5, n_f - 0.5)
    ax.set_ylim(0, fam.loss.max())
    for s in ('left', 'top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(range(n_f), [COLUMN_LABEL[f] for f in fam.index], rotation=90, linespacing=1.0)
    ax.tick_params(axis='x', length=0)
    letter(0.0, 1.42, 'D')
    fig.text(0.22 / W, 1.42 / H, 'Accuracy loss when each family is permuted (%)', ha='left', va='top',
             fontsize=st.AXIS_LABEL)

    # ---------------- E: change in recall per family (rows, in the order of D) and mechanism (columns) (fig 7)
    rc = pd.read_csv(PERM / 'recall_change.csv')
    M = rc.pivot(index='unit', columns='mech', values='change_points').loc[fam.index, C.MECHS]
    v = M.to_numpy()
    e_left, e_right, e_top, e_bot = 4.7, 8.45, 3.08, 0.82
    ax = box(e_left, e_bot, e_right - e_left, e_top - e_bot)
    im = ax.imshow(v, cmap=st.DIVERGING, vmin=-70, vmax=70, aspect='auto')
    ax.set_xticks(np.arange(21) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_f + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(20), C.MECHS, rotation=90)
    ax.set_yticks(range(n_f), [st.FAMILY_LABEL[f] for f in fam.index])
    ax.set_xlabel('Mechanism', labelpad=2)
    for _, a, _b in C.CATEGORIES[1:]:
        ax.axvline(a - 0.5, color=st.INK, lw=0.6)
    for i in range(n_f):
        for j in range(20):
            if abs(v[i, j]) >= 5:
                ax.text(j, i, ('+%d' if v[i, j] > 0 else '−%d') % round(abs(v[i, j])), ha='center',
                        va='center', fontsize=st.CELL, color='white' if abs(v[i, j]) >= 40 else st.INK)
    st.category_brackets(ax, where=1.025)
    cax = box(e_right + 0.07, e_bot, 0.06, e_top - e_bot)
    cb = fig.colorbar(im, cax=cax, ticks=[-60, -30, 0, 30, 60])
    cb.set_label('Change in recall (%)', labelpad=2)
    cb.outline.set_linewidth(0.4)
    letter(3.62, row, 'E')
    st.save(fig, C.REPORT, 'Figure_2_models_and_permutation')


# ----------------------------------------------------------------------------------------------- EDA pictures
def class_balance():
    """Share of every mechanism in the training, validation and test sets."""
    B = pd.read_csv(C.TABLES / 'eda' / 'class_balance.csv', index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 2.3), sharey=True, layout='constrained')
    x = np.arange(20)
    for ax, (col, name) in zip(axes, (('train', 'Training'), ('val', 'Validation'), ('test', 'Test'))):
        share = 100 * B[col] / B[col].sum()
        ax.bar(x, share, width=0.7, color=st.BLUE)
        ax.axhline(5, color=st.MUTED, lw=0.6, ls='--')
        ax.set_title('%s (%s experiments)' % (name, format(int(B[col].sum()), ',')), fontsize=st.PANEL_TITLE, loc='left')
        ax.set_xticks(x, C.MECHS, rotation=90)
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(0, 6)
        print('%s: %.2f-%.2f%% per mechanism' % (name, share.min(), share.max()))
    axes[0].set_ylabel('Share of experiments (%)')
    st.save(fig, OUT_EDA, 'class_balance')


def data_quality():
    """The data-quality checks as a table: missing, infinite and negative values and identical experiments."""
    Q = pd.read_csv(C.TABLES / 'eda' / 'data_quality.csv')
    names = {'x1_train (subset)': 'Training loadings, x1', 'x2_train (subset)': 'Training runs, x2',
             'y_train (subset)': 'Training labels, y'}
    rows = []
    for r in Q.itertuples():
        name = names.get(r.array, r.array.replace('x2_test, ', 'Test runs, '))
        dup = '' if pd.isna(r.identical_experiments) else '%d' % r.identical_experiments
        shape = ' x '.join(format(int(v), ',') for v in r.shape.strip('(),').split(',') if v.strip())
        rows.append([name, shape, '%d' % r.NaN, '%d' % r.inf, format(int(r.negative), ','),
                     ('%.3g' % r.min).replace('-', '−'), '%.3g' % r.max, dup])
    header = ['Array', 'Shape', 'Missing', 'Infinite', 'Negative', 'Min', 'Max', 'Identical\nexperiments']
    fig, ax = plt.subplots(figsize=(st.FULL, 0.28 * (len(rows) + 1) + 0.5), layout='constrained')
    ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=header, loc='upper center', cellLoc='right', colLoc='right',
                   colWidths=[0.27, 0.14, 0.08, 0.08, 0.09, 0.11, 0.08, 0.12])
    tab.auto_set_font_size(False)
    tab.set_fontsize(st.TICK)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor('#d9d8d4')
        cell.set_linewidth(0.4)
        cell.set_height(0.9 / (len(rows) + 1))
        if j == 0:
            cell._loc = 'left'
            cell.get_text().set_ha('left')
        if i == 0:
            cell.get_text().set_fontweight('bold')
            cell.set_facecolor('#f0efec')
    ax.text(0, -0.02, 'Training arrays: a random subset of 100,000 experiments. Small negative values are numerical '
            '(about $-10^{-13}$) or the noise added to the test sets.', transform=ax.transAxes, fontsize=st.ANNOT,
            va='top', color=st.MUTED)
    st.save(fig, OUT_EDA, 'data_quality')


REPORT_FIGS = {'figure1': figure1_feature_families, 'figure2': figure2_models_and_permutation,
               'appendix3': appendix3_feature_distributions, 'appendix4': appendix4_pearson_correlation}
SUPPORTING_FIGS = {'pearson_square': supporting_pearson_square, 'example_run': supporting_example_run,
                   'models': supporting_models, 'confusion': supporting_confusion,
                   'family_accuracy': supporting_family_accuracy, 'family_recall': supporting_family_recall,
                   'family_breadth': supporting_family_breadth, 'balance': class_balance, 'quality': data_quality}
FIGS = {**REPORT_FIGS, **SUPPORTING_FIGS}


def main():
    keys = []
    for k in sys.argv[1:] or ['report', 'supporting']:
        keys += list(REPORT_FIGS) if k == 'report' else list(SUPPORTING_FIGS) if k == 'supporting' else [k]
    for k in keys:
        FIGS[k]()
        print('%s done' % k, flush=True)


if __name__ == '__main__':
    main()
