"""Figure style shared by every figure: one font, one text size per region, fixed figure widths.

Figures are saved at their final printed size (no cropping), so a text region (axis labels, tick labels, legends,
labels inside a plot, numbers in heatmap cells) has the same point size in every figure when the files are placed
in the paper at 100% scale. Letter paper with 1 in margins: FULL = 6.5 in, HALF = 3.15 in.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402

from . import config as C                                         # noqa: E402

FULL, HALF = 6.5, 3.15

# text size (pt) by region
AXIS_LABEL = 9        # axis titles, colour-bar titles
TICK = 8              # tick labels, family names along an axis
LEGEND = 8
PANEL_TITLE = 9       # title above one panel of a multi-panel figure
ANNOT = 7.5           # labels placed inside a plot: point names, bar values
CELL = 6.5            # numbers inside heatmap cells
PANEL_LETTER = 10     # bold A, B, C ... of a lettered multi-panel figure

INK, MUTED, RULE, GRID = '#1a1a1a', '#555555', '#8c8a85', '#e4e3df'
BLUE, ORANGE, GREEN, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948'
LIGHT_BLUE = '#a9c9ef'
DIVERGING = LinearSegmentedColormap.from_list('red_grey_blue', [RED, '#f0efec', BLUE])
SEQUENTIAL = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cfe0f5', BLUE, '#0e3f7a'])

# names used in the paper's figures
FAMILY_LABEL = {'Pooled order (VTNA)': 'VTNA', 'Catalyst order (run pairs)': 'Catalyst order',
                'Same-excess test': 'Same-excess test', 'Selwyn test': 'Selwyn test',
                'Integrated fit (MM)': 'MM fit', 'Death laws': 'Deactivation laws', 'Inhibitor laws': 'Inhibitor laws',
                'Mass deficit': 'Mass deficit', 'Deficit Km fit': 'Deficit $K_\\mathrm{m}$ fit', 'Curve shapes': 'Curve shapes',
                'Catalyst loadings': 'Catalyst loadings', 'Noise and sampling': 'Noise and sampling'}


def bold_math(s):
    """The same label for bold text: math (the subscripted K_m) in bold too."""
    return s.replace(r'$K_\mathrm{m}$', r'$\boldsymbol{K}_\mathbf{m}$')


def use():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': TICK,
        'axes.labelsize': AXIS_LABEL, 'axes.titlesize': PANEL_TITLE, 'axes.titleweight': 'normal',
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'legend.fontsize': LEGEND, 'legend.frameon': False,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#333333', 'xtick.color': '#333333',
        'ytick.color': '#333333', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
        'figure.constrained_layout.h_pad': 0.03, 'figure.constrained_layout.w_pad': 0.03,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300,
    })


def save(fig, out, name):
    """PDF (vector, fonts embedded) and PNG (300 dpi), both at the figure's exact size."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / ('%s.%s' % (name, ext)))
    plt.close(fig)


def category_brackets(ax, where=1.02, size=TICK, lw=0.6):
    """Core / Bicatalytic / Activation / Deactivation brackets above an M1-M20 x axis."""
    tr = ax.get_xaxis_transform()
    for name, a, b in C.CATEGORIES:
        ax.plot([a - 0.4, b - 0.6], [where, where], color=INK, lw=lw, transform=tr, clip_on=False)
        ax.text((a + b - 1) / 2, where + 0.012, name, transform=tr, ha='center', va='bottom', fontsize=size)
