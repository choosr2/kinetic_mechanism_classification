"""Step 7: the report's Appendices 5 and 6 (Word) and a supporting document.

  results/report/Appendix_5_hyperparameter_optimisation.docx
      report Appendix 5, Tables 5A-5E: every hyperparameter setting tried, one table per model
      (results/tables/search.csv, winner.json)
  results/report/Appendix_6_engineered_features.docx (and .csv)
      report Appendix 6, Tables 6A-6L: all 136 features and what each measures, one table per family; features removed
      by the correlation filter are marked with a dagger
  results/supporting/Supporting - Engineered features by family.docx   (not in the report)
      the same 136 features with the source paper of each method, and one example figure per family (Figures S1-S12,
      results/supporting/feature_examples/, from plots_appendix.py)
The "Death laws" family of the code is called "Deactivation laws", as in the report.

Usage:  uv run python -m kinetic_mech.appendix [5 6 supporting]      (default: all)
"""
import json
import math
import re
from html.parser import HTMLParser

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

from . import config as C
from .features import COLUMNS, families, keep_columns

OUT_A = C.SUPPORTING / 'Supporting - Engineered features by family.docx'
OUT_B = C.REPORT / 'Appendix_5_hyperparameter_optimisation.docx'
OUT_6 = C.REPORT / 'Appendix_6_engineered_features.docx'
FIGS = C.SUPPORTING / 'feature_examples'
BLOCK_OF = {f: b for b, fs in (('Graphical tests', C.FAMILY_ORDER[:4]), ('Integrated rate-law fits', C.FAMILY_ORDER[4:7]),
                               ('Mass balance', C.FAMILY_ORDER[7:9]), ('Not in a block', C.FAMILY_ORDER[9:]))
            for f in fs}
BLOCK_TITLE = {'Graphical tests': 'Graphical tests', 'Integrated rate-law fits': 'Integrated rate-law fits',
               'Mass balance': 'Mass balance', 'Not in a block': 'Context (not a kinetic test)'}


def pct(x, d=2):
    return '%.*f' % (d, 100 * x)


def fmt_num(v, sig=3):
    """A number to `sig` significant figures, without trailing zeros."""
    if v == 0:
        return '0'
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return ('%.*f' % (d, v)).rstrip('0').rstrip('.') if d else '%d' % round(v)


# ================================================================== what each feature means, and where it comes from
LAWS = ['free', 'sub', 'sub2', 'pfree', 'psub', 'free_b', 'sub_b']
RUNS = [1, 2, 3, 4]
WORK = 'This work'

# key -> (short label used in the Source column and notes, full reference as HTML)
REFS = {
    'blackmond2005': ('Blackmond 2005', 'Blackmond DG (2005). Reaction progress kinetic analysis: a powerful methodology '
                      'for mechanistic studies of complex catalytic reactions. <i>Angew. Chem. Int. Ed.</i> 44, '
                      '4302&ndash;4320. doi:10.1002/anie.200462544'),
    'blackmond2015': ('Blackmond 2015', 'Blackmond DG (2015). Kinetic profiling of catalytic organic reactions as a '
                      'mechanistic tool. <i>J. Am. Chem. Soc.</i> 137, 10852&ndash;10866. doi:10.1021/jacs.5b05841'),
    'bonvin2015': ('Bonvin group 2015', 'Bonvin group, EPFL (2015). Reconciliation of noisy reaction concentrations '
                   'using reaction extents. ESCAPE 25, <i>Comput. Aided Chem. Eng.</i> 37. '
                   'infoscience.epfl.ch/record/201513 [authors and exact title to be confirmed]'),
    'bures2016a': ('Bur&eacute;s 2016a', 'Bur&eacute;s J (2016a). A simple graphical method to determine the order in '
                   'catalyst. <i>Angew. Chem. Int. Ed.</i> 55, 2028&ndash;2031. doi:10.1002/anie.201508983'),
    'bures2023': ('Bur&eacute;s &amp; Larrosa 2023', 'Bur&eacute;s J, Larrosa I (2023). Organic reaction mechanism '
                  'classification using machine learning. <i>Nature</i> 613, 689&ndash;695. '
                  'doi:10.1038/s41586-022-05639-4'),
    'dalland2024': ('Dalland et al. 2024', 'Dalland D, Schrecker L, Hii KK (2024). Auto-VTNA: an automatic VTNA platform '
                    'for determination of global rate laws. <i>Digital Discovery</i> 3, 2118&ndash;2129. '
                    'doi:10.1039/d4dd00111g'),
    'duggleby1986': ('Duggleby 1986', 'Duggleby RG (1986). Progress curves of reactions catalyzed by unstable enzymes. A '
                     'theoretical approach. <i>J. Theor. Biol.</i> 123, 67&ndash;80. doi:10.1016/s0022-5193(86)80236-3'),
    'duggleby1995': ('Duggleby 1995', 'Duggleby RG (1995). Analysis of enzyme progress curves by nonlinear regression. '
                     '<i>Methods Enzymol.</i> 249, 61&ndash;90. doi:10.1016/0076-6879(95)49031-0'),
    'goudar1999': ('Goudar et al. 1999', 'Goudar CT, Sonnad JR, Duggleby RG (1999). Parameter estimation using a direct '
                   'solution of the integrated Michaelis&ndash;Menten equation. <i>Biochim. Biophys. Acta</i> 1429, '
                   '377&ndash;383. doi:10.1016/s0167-4838(98)00247-7'),
    'martinez2019': ('Mart&iacute;nez-Carri&oacute;n et al. 2019', 'Mart&iacute;nez-Carri&oacute;n A, Howlett MG, '
                     'Alamillo-Ferrer C, et al. (2019). Kinetic treatments for catalyst activation and deactivation '
                     'processes based on variable time normalization analysis. <i>Angew. Chem. Int. Ed.</i> 58, '
                     '10189&ndash;10193. doi:10.1002/anie.201903878'),
    'nielsen2019': ('Nielsen &amp; Bur&eacute;s 2019', 'Nielsen CDT, Bur&eacute;s J (2019). Visual kinetic analysis. '
                    '<i>Chem. Sci.</i> 10, 348&ndash;353. doi:10.1039/c8sc04698k'),
    'schnell1997': ('Schnell &amp; Mendoza 1997', 'Schnell S, Mendoza C (1997). Closed form solution for time-dependent '
                    'enzyme kinetics. <i>J. Theor. Biol.</i> 187, 207&ndash;212. doi:10.1006/jtbi.1997.0425'),
    'seltzer2013': ('Seltzer et al. 2013', 'Seltzer ML, Yu D, Wang Y (2013). An investigation of deep neural networks for '
                    'noise robust speech recognition. <i>Proc. IEEE ICASSP 2013</i>, 7398&ndash;7402. '
                    'doi:10.1109/icassp.2013.6639100'),
    'selwyn1965': ('Selwyn 1965', 'Selwyn MJ (1965). A simple test for inactivation of an enzyme during assay. '
                   '<i>Biochim. Biophys. Acta</i> 105, 193&ndash;195. doi:10.1016/s0926-6593(65)80190-4'),
    'zhang2017': ('Zhang et al. 2017', 'Zhang K, Zuo W, Zhang L (2017). FFDNet: toward a fast and flexible solution for '
                  'CNN-based image denoising. arXiv:1710.04026'),
}


def cols(pattern, values):
    return [pattern % v for v in values]


def cite(key):
    return REFS[key][0]


# family -> (note shown under the heading, [(columns, meaning, sources), ...]); meanings and notes are HTML;
# sources are REFS keys or WORK, shown in order
DESCRIPTIONS = {
    'Pooled order (VTNA)': (
        '<code>_all</code> uses conversion from 5% up to the highest level all three runs reach; '
        '<code>_core</code> uses only 15&ndash;85%.',
        [(['vtna_order_all', 'vtna_order_core'], 'The single catalyst order that best lines up runs 1&ndash;3 when time '
                                                 'is scaled by [cat]<sub>0</sub><sup>order</sup>, fitted at 8 conversion '
                                                 'levels', ['bures2016a']),
         (['vtna_rms_all', 'vtna_rms_core'], 'How far apart the runs still are at that best order (RMS residual of '
                                             'ln t)', ['dalland2024']),
         (['vtna_rms_noise_all', 'vtna_rms_noise_core'], 'The same residual divided by <code>noise_est</code>: is the '
                                                         'mismatch bigger than the measurement noise?',
          ['dalland2024', WORK]),
         (['vtna_width_all', 'vtna_width_core'], 'The range of orders that fit almost as well (within 15% of the best '
                                                 'error). Wide means the order is poorly defined', ['dalland2024']),
         (['vtna_res_low_all', 'vtna_res_low_core'], 'Mean residual of run 1, the lowest loading', [WORK]),
         (['vtna_res_high_all', 'vtna_res_high_core'], 'Mean residual of run 3, the highest loading. Together with '
                                                       '<code>res_low</code>, it shows when the rate does not follow a '
                                                       'single power of catalyst (dimers, or a poison that removes a '
                                                       'fixed amount)', [WORK])]),
    'Catalyst order (run pairs)': (
        '',
        [(['cat_order_12', 'cat_order_23', 'cat_order_13'], 'Apparent order in catalyst, from how much faster the '
                                                            'higher-loading run makes 0.2 of product (run 1 vs 2, run 2 '
                                                            'vs 3, run 1 vs 3)', ['bures2016a']),
         (['X_cat_order_12', 'X_cat_order_23', 'X_cat_order_13'], 'The same, read from pooled conversion X',
          ['bures2016a'])]),
    'Same-excess test': (
        'The test is published as a pair of plotted curves; the numeric scores are from this work.',
        [(['same_excess_log_ratio'], 'ln(time run 4 takes to make 0.1 of product &divide; time the matching run takes '
                                     'over the same stretch). 0 means the runs line up; &lt; 0 means run 4 is faster '
                                     '(catalyst lost in the matching run); &gt; 0 means run 4 is slower (catalyst '
                                     'activating)', ['blackmond2005', 'nielsen2019']),
         (['X_same_excess_log_ratio'], 'The same, read from pooled conversion X', ['blackmond2005', 'nielsen2019']),
         (['same_excess_mean'], 'Mean conversion gap between the time-shifted matching run and run 4 over the whole '
                                'run. &lt; 0 means run 4 is ahead', ['blackmond2005', 'nielsen2019']),
         (['same_excess_gap_5_20', 'same_excess_gap_20_35'], 'ln(matching run&rsquo;s time &divide; run 4&rsquo;s time) to '
                                                             'go from 5 to 20%, and from 20 to 35%, above run 4&rsquo;s '
                                                             'start. &gt; 0 means run 4 is faster. Both windows start '
                                                             'after the initial burst while the catalyst binds substrate',
          ['blackmond2005', 'nielsen2019'])]),
    'Selwyn test': (
        'The test is published as a plot; the numeric scores are from this work.',
        [(['selwyn_12', 'selwyn_13', 'selwyn_23'], 'Mean conversion gap between the lower- and higher-loading run '
                                                   'plotted against [cat]<sub>0</sub>&middot;t (run 1 vs 2, run 1 vs 3, '
                                                   'run 2 vs 3). 0 means the curves overlap, as for a stable catalyst '
                                                   'that is first order', ['selwyn1965', 'nielsen2019']),
         (['selwyn_rms'], 'RMS spread of all three runs on the [cat]<sub>0</sub>&middot;t axis',
          ['selwyn1965', 'nielsen2019'])]),
    'Integrated fit (MM)': (
        'Fits [cat]<sub>0</sub>&middot;t = a(S<sub>0</sub> &minus; S) + b ln(S<sub>0</sub>/S) to runs 1&ndash;3 together, '
        'because fits to single curves are unreliable at 1%% noise (%s).' % cite('duggleby1995'),
        [(['mm_km'], 'b/a, a K<sub>m</sub>-like shape number. It is often negative on this data, probably because of '
                     'the reversible step, so treat it as a shape descriptor, not a physical K<sub>m</sub>',
          ['schnell1997', 'goudar1999']),
         (['mm_sat'], 'Share of the time to half conversion that comes from the zero-order term a. Near 1, the catalyst '
                      'stays saturated; near 0, the run behaves as first order', [WORK]),
         (['mm_a'], 'Coefficient a, roughly 1/k<sub>cat</sub> (the zero-order part), scaled',
          ['schnell1997', 'goudar1999']),
         (['mm_b'], 'Coefficient b, roughly K<sub>m</sub>/k<sub>cat</sub> (the first-order part), scaled',
          ['schnell1997', 'goudar1999']),
         (['mm_rmse'], 'How well the law fits runs 1&ndash;3', [WORK]),
         (['mm_res_1', 'mm_res_2', 'mm_res_3'], 'Mean residual in each fitted run: where the plain law misses', [WORK]),
         (['mm_res_4'], 'Mean residual in run 4, which the fit never saw', [WORK])]),
    'Death laws': (
        'Each adds one catalyst-death term to the Michaelis&ndash;Menten fit. The terms come from dividing the death '
        'rate by the turnover rate, which gives live catalyst as a function of substrate left; %s (eq. 5) gives this '
        'for M9, M14 and M20, and the other laws were derived the same way in this work. The laws: <b>free</b>, the '
        'catalyst dies on its own (M9); <b>sub</b>, loss tracks the substrate used (M11, M14); <b>sub2</b>, the '
        'catalyst&ndash;substrate complex dies with a second substrate (M16); <b>pfree</b>, <b>psub</b>, death with '
        'product (M12, M17); <b>free_b</b>, <b>sub_b</b>, pair deaths, where the fraction lost does not depend on '
        'loading (M13, M18, M19).' % cite('duggleby1986'),
        [(cols('law_rel_%s', LAWS), 'How much worse that law fits than the best law (0 means it is the best)',
          ['duggleby1986', 'martinez2019']),
         (cols('law_loss_%s', LAWS), 'How much catalyst the law says was lost. For the five laws that scale with '
                                     'loading (all but the pair deaths), it is relative to run 1&rsquo;s loading',
          ['duggleby1986', 'martinez2019']),
         (cols('law_res4_%s', LAWS), 'Mean residual in run 4 under that law: does the death law predict the '
                                     'same-excess run?', ['duggleby1986', WORK]),
         (['law_best'], 'Which of the 7 laws fits best (a category)', ['duggleby1986', 'martinez2019'])]),
    'Inhibitor laws': (
        'Instead of a death term, these fits add either a poison present from the start or an inhibitor that gets '
        'used up. The <b>free</b> form is an inhibitor that binds free catalyst (M10); the <b>sub</b> form binds the '
        'catalyst&ndash;substrate complex (M15). Each has a shape parameter g of 0.5, 2 or 8. The used-up inhibitor '
        'terms are derived with the same step as the death laws (%s).' % cite('duggleby1986'),
        [(['inh_gain_poison'], 'Share of the plain fit&rsquo;s error removed by a poison that takes out a fixed amount '
                               'of catalyst from the start', [WORK]),
         (['inh_amount_poison'], 'That poison amount, relative to run 1&rsquo;s loading', [WORK]),
         (['inh_res4_poison'], 'Mean residual in run 4 under the poison fit', [WORK]),
         (['inh_gain_free_0.5', 'inh_gain_free_2', 'inh_gain_free_8'], 'Share of error removed by a used-up inhibitor '
                                                                       'on free catalyst, at each shape g',
          [WORK, 'duggleby1986']),
         (['inh_gain_sub_0.5', 'inh_gain_sub_2', 'inh_gain_sub_8'], 'The same for an inhibitor on the '
                                                                    'catalyst&ndash;substrate complex',
          [WORK, 'duggleby1986']),
         (['inh_best_free', 'inh_best_sub'], 'Which shape (g = 0.5, 2 or 8) fits best for each form (a category)',
          [WORK, 'duggleby1986']),
         (['inh_amount_free', 'inh_amount_sub'], 'Inhibitor amount at the best shape, relative to run 1&rsquo;s loading',
          [WORK, 'duggleby1986']),
         (['inh_res4_free', 'inh_res4_sub'], 'Mean residual in run 4 at the best shape', [WORK, 'duggleby1986']),
         (['inh_vs_law'], 'Best inhibitor fit&rsquo;s error &divide; best death-law fit&rsquo;s error. &lt; 1 means an '
                          'inhibitor explains the runs better than catalyst death', [WORK])]),
    'Mass deficit': (
        'The deficit D = 1 &minus; S &minus; P is material missing from solution, i.e. held on the catalyst. It is '
        'averaged over the last quarter of each run. No published source was found for this test.',
        [(['deficit_slope'], 'Slope of deficit against loading across runs 1&ndash;3, through the origin: material '
                             'bound per unit of catalyst', [WORK]),
         (cols('deficit_per_cat_%d', RUNS), 'Deficit &divide; loading for each run', [WORK])]),
    'Deficit Km fit': (
        'Fits D&middot;S = &epsilon;&middot;[cat]<sub>0</sub>&middot;S &minus; K<sub>m</sub>&middot;D to runs 1&ndash;3. '
        'This reads substrate binding from the deficit rather than from the curve shape. The equation is from this '
        'work.',
        [(['dk_km1'], 'K<sub>m</sub> from the deficit, assuming all the catalyst can bind (&epsilon; = 1)', [WORK]),
         (['dk_km2'], 'K<sub>m</sub> with &epsilon; fitted as well', [WORK]),
         (['dk_eps'], '&epsilon;, the fraction of catalyst that binds substrate', [WORK]),
         (['dk_rmse'], 'How well the deficit equation fits runs 1&ndash;3', [WORK]),
         (['dk_res_4'], 'Mean residual of the deficit equation in run 4', [WORK]),
         (['dk_km1_early', 'dk_km1_late'], '<code>dk_km1</code> from points with S &ge; 0.5 and S &lt; 0.5: does '
                                           'binding change during the run?', [WORK]),
         (['dk_gap1', 'dk_gap2'], '<code>dk_km1</code> and <code>dk_km2</code> minus <code>mm_km</code>: do binding '
                                  'and curve shape agree on K<sub>m</sub>?', [WORK, 'schnell1997']),
         (['dk_logratio'], 'ln(<code>dk_km1</code> / <code>mm_km</code>), when both are positive', [WORK, 'schnell1997']),
         (['dk_sat'], 'Catalyst saturation at S = 0.5 implied by <code>dk_km1</code>, i.e. 0.5/(K<sub>m</sub> + 0.5)',
          [WORK]),
         (['dk_sat_gap'], '<code>dk_sat</code> minus the saturation implied by the Michaelis&ndash;Menten curve fit '
                          '(<code>mm_sat</code>)', [WORK, 'schnell1997'])]),
    'Curve shapes': (
        'Read from product alone, or from pooled conversion X for the <code>X_</code> versions. Reading the shape of '
        'whole progress curves follows reaction progress kinetic analysis (%s, %s); the specific measures are from '
        'this work.' % (cite('blackmond2005'), cite('blackmond2015')),
        [(cols('P_end_%d', RUNS), 'Product at the last measured point of that run: how far it got', [WORK]),
         (cols('X_P_end_%d', RUNS), 'Pooled conversion X at the last measured point', [WORK]),
         (cols('t_end_%d', RUNS), 'Time of the last measured point (run length)', [WORK]),
         (cols('shape25_%d', RUNS), 'Share of the run&rsquo;s product gain reached by a quarter of its duration. High '
                                    'means a fast start then a stall (e.g. catalyst death); low means a slow start '
                                    '(e.g. activation)', [WORK]),
         (cols('shape50_%d', RUNS), 'The same, by half of its duration', [WORK]),
         (cols('t_half_frac_%d', RUNS), 'Time to reach half of the run&rsquo;s product gain, as a share of its length. '
                                        'Low means a fast start', [WORK]),
         (cols('X_shape25_%d', RUNS), 'Share of the run&rsquo;s gain in X reached by a quarter of its duration', [WORK]),
         (cols('X_shape50_%d', RUNS), 'The same, by half of its duration', [WORK]),
         (cols('X_t_half_frac_%d', RUNS), 'Time to reach half of the run&rsquo;s gain in X, as a share of its length',
          [WORK])]),
    'Catalyst loadings': (
        '',
        [(['cat0_1', 'cat0_2', 'cat0_3'], 'Catalyst loading of runs 1&ndash;3 (an input, not a test)', ['bures2023']),
         (['cat_spread'], 'ln(<code>cat0_3</code> / <code>cat0_1</code>): how far apart the loadings are', [WORK])]),
    'Noise and sampling': (
        '',
        [(['noise_est'], 'Measurement noise, from the scatter of S + P within each run (S + P should stay nearly '
                         'constant). Giving a model the noise level as an input follows speech and image denoising',
          [WORK, 'seltzer2013', 'zhang2017']),
         (cols('max_gap_%d', RUNS), 'Longest gap between time points, as a share of the run&rsquo;s length', [WORK])]),
}


def check(all_cols, fam_all):
    """Every column in exactly one row of its own family; every source a known reference or WORK."""
    seen = []
    for f in C.FAMILY_ORDER:
        listed = [c for cs, _, _ in DESCRIPTIONS[f][1] for c in cs]
        if sorted(listed) != sorted(fam_all[f]):
            raise SystemExit('%s: listed %s, actual %s' % (f, sorted(listed), sorted(fam_all[f])))
        seen += listed
        for cs, _, src in DESCRIPTIONS[f][1]:
            bad = [s for s in src if s != WORK and s not in REFS]
            if not src or bad:
                raise SystemExit('%s: missing or unknown source %s' % (cs[0], bad))
    if len(seen) != len(set(seen)) or set(seen) != set(all_cols):
        raise SystemExit('rows do not cover the columns exactly once')


def source_cell(src):
    return '; '.join(WORK if s == WORK else cite(s) for s in src)



# ================================================================== Word documents
FONT, MONO = 'Times New Roman', 'Consolas'
GREY = RGBColor(0x66, 0x66, 0x66)
TEXT_W = 6.5                              # in: Letter with 1 in margins

FAMILY_NAME = {f: f for f in C.FAMILY_ORDER}
FAMILY_NAME['Death laws'] = 'Deactivation laws'


def deactivation(text):
    """The paper calls the death laws deactivation laws."""
    text = re.sub(r'\bDeath(?=[ -]laws?\b)', 'Deactivation', text)
    text = re.sub(r'\bdeath(?=[ -]laws?\b)', 'deactivation', text)
    for a, b in (('catalyst-death term', 'catalyst-deactivation term'), ('the death rate', 'the deactivation rate'),
                 ('Instead of a death term', 'Instead of a deactivation term'),
                 ('better than catalyst death', 'better than catalyst deactivation')):
        text = text.replace(a, b)
    return text


# ------------------------------------------------------------------------------------------------ helpers
class Runs(HTMLParser):
    """Turn the small HTML used in the descriptions (b, i, code, sub, sup, br) into Word runs."""

    def __init__(self, par, size, color=None):
        super().__init__(convert_charrefs=True)
        self.par, self.size, self.color, self.stack = par, size, color, []

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.par.add_run().add_break()
        else:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            self.stack.reverse()
            self.stack.remove(tag)
            self.stack.reverse()

    def handle_data(self, data):
        r = self.par.add_run(data)
        s = set(self.stack)
        r.bold = 'b' in s or None
        r.italic = 'i' in s or None
        r.font.subscript = 'sub' in s or None
        r.font.superscript = 'sup' in s or None
        if 'code' in s:
            r.font.name = MONO
            r.font.size = Pt(self.size - 1.5)
        else:
            r.font.size = Pt(self.size)
        if self.color is not None:
            r.font.color.rgb = self.color


def rich(par, text, size=11, color=None):
    p = Runs(par, size, color)
    p.feed(text)
    p.close()
    return par


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Inches(1))
    st = doc.styles['Normal']
    st.font.name, st.font.size = FONT, Pt(11)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    for name, size, before, after in (('Title', 16, 0, 6), ('Heading 1', 13, 14, 4), ('Heading 2', 11.5, 12, 4)):
        h = doc.styles[name]
        h.font.name, h.font.size, h.font.bold, h.font.italic = FONT, Pt(size), True, False
        h.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        rpr = h.element.get_or_add_rPr()
        fonts = rpr.find(qn('w:rFonts'))
        if fonts is None:
            fonts = OxmlElement('w:rFonts')
            rpr.append(fonts)
        for a in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
            fonts.set(qn(a), FONT)
        for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
            if fonts.get(qn(a)) is not None:
                del fonts.attrib[qn(a)]
        pf = h.paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), True
        if name == 'Title':
            ppr = h.element.get_or_add_pPr()
            for b in ppr.findall(qn('w:pBdr')):
                ppr.remove(b)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(footer.add_run(), 'PAGE', 9)


def field(run, code, size):
    run.font.size = Pt(size)
    for kind, text in (('begin', None), (None, code), ('end', None)):
        if kind:
            e = OxmlElement('w:fldChar')
            e.set(qn('w:fldCharType'), kind)
        else:
            e = OxmlElement('w:instrText')
            e.set(qn('xml:space'), 'preserve')
            e.text = text
        run._r.append(e)


def borders(cell, **edges):
    """edges: top/bottom/left/right = (size in eighths of a point, colour hex) or None for no border."""
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders')
        tcPr.append(b)
    for side in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement('w:' + side)
        spec = edges.get(side)
        if spec:
            e.set(qn('w:val'), 'single')
            e.set(qn('w:sz'), str(spec[0]))
            e.set(qn('w:color'), spec[1])
        else:
            e.set(qn('w:val'), 'nil')
        b.append(e)


def margins(cell, top=30, bottom=30, left=0, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    m = OxmlElement('w:tcMar')
    for side, v in (('top', top), ('bottom', bottom), ('left', left), ('right', right)):
        e = OxmlElement('w:' + side)
        e.set(qn('w:w'), str(v))
        e.set(qn('w:type'), 'dxa')
        m.append(e)
    tcPr.append(m)


def row_flags(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement('w:cantSplit')
    trPr.append(e)
    if header:
        e = OxmlElement('w:tblHeader')
        trPr.append(e)


def table(doc, header, rows, widths, align=None, size=9.5, bold_rows=(), rule_before=()):
    """Booktabs-style table: rule above and below the header, light rules between rows, rule at the bottom.
    rows: lists of cell contents; a cell is HTML text or a callable(paragraph) that fills it."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement('w:tblLayout')
    lay.set(qn('w:type'), 'fixed')
    tblPr.append(lay)
    align = align or ['l'] * len(header)
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate([header] + rows):
        tr = t.rows[i]
        row_flags(tr, header=i == 0)
        last = i == len(rows)
        for j, val in enumerate(row):
            c = tr.cells[j]
            c.width = Inches(widths[j])
            margins(c)
            top = (8, '222222') if i == 0 else ((6, '222222') if i in rule_before else None)
            bottom = (6, '222222') if i == 0 or last else (2, 'BBBBBB')
            borders(c, top=top, bottom=bottom)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i == 0          # never leave a header alone at a page foot
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if align[j] == 'r' else WD_ALIGN_PARAGRAPH.LEFT
            if callable(val):
                val(p)
            else:
                rich(p, '<b>%s</b>' % val if i == 0 or i in bold_rows else val, size)
    return t


def caption(doc, label, text, keep_next=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = keep_next
    r = p.add_run(label + ' ')
    r.bold, r.font.size = True, Pt(9.5)
    rich(p, text, 9.5)
    return p


# ------------------------------------------------------------------------------------------------ appendix A
FIG_CAPTION = {
    'Pooled order (VTNA)': 'Pooled order (VTNA) on a typical M4 experiment. (a) Conversion X = (1 &minus; S + P)/2 '
        'against time for runs 1&ndash;3. (b) Time multiplied by [cat]<sub>0</sub> (order 1): runs 2 and 3 still lie '
        'above run 1. (c) Time multiplied by [cat]<sub>0</sub><sup>n</sup> at the fitted order n = 1.56 '
        '(<code>vtna_order_all</code>), where the three runs line up. <code>vtna_rms</code> is the spread left in (c); '
        '<code>vtna_res_low</code> and <code>vtna_res_high</code> are the mean residuals of runs 1 and 3.',
    'Catalyst order (run pairs)': 'Catalyst order from pairs of runs, same M4 experiment. (a) Product against time; '
        'dotted lines mark when each run has made 0.2 of product. (b) ln t(P = 0.2) against ln [cat]<sub>0</sub>; the '
        'order of each pair of runs is minus the slope between its two points.',
    'Same-excess test': 'Same-excess test. The matching run (the run of 1&ndash;3 with run 4&rsquo;s loading, grey) is '
        'shifted in time so that it starts at run 4&rsquo;s product level, while run 4 starts with fresh catalyst. '
        '(a) M1, stable catalyst: the curves line up. (b) M14, catalyst lost during the matching run: run 4 is faster '
        '(log ratio &lt; 0). (c) M6, catalyst activating: run 4 is slower (log ratio &gt; 0).',
    'Selwyn test': 'Selwyn test: conversion against [cat]<sub>0</sub>&middot;t for runs 1&ndash;3. (a) M1: a stable '
        'catalyst that is first order gives a single curve. (b) M6 (activation): the curves separate.',
    'Integrated fit (MM)': 'Integrated Michaelis&ndash;Menten fit, [cat]<sub>0</sub>&middot;t = a(S<sub>0</sub> &minus; '
        'S) + b ln(S<sub>0</sub>/S), fitted to runs 1&ndash;3 (solid) and used to predict the held-out run 4 (dashed). '
        '(a) M1 follows the law. (b) M9, whose catalyst deactivates, departs from it at high conversion.',
    'Death laws': 'Deactivation laws on a typical M12 experiment (catalyst deactivated by product). (a) The best of '
        'the seven laws (<b>pfree</b>, deactivation with product), fitted to runs 1&ndash;3 (solid, one curve per '
        'loading) and predicting run 4 (dashed). (b) <code>law_rel</code> of each law: how much worse it fits than '
        'the best law (0 = best).',
    'Inhibitor laws': 'Inhibitor laws on a typical M10 experiment (inhibitor binding free catalyst). (a) The poison fit, '
        'in which a fixed amount of catalyst is lost from the start (<code>inh_amount_poison</code> = 0.58 of run '
        '1&rsquo;s loading). (b) Share of the plain Michaelis&ndash;Menten fit&rsquo;s error removed by the poison law '
        'and by the used-up-inhibitor laws at each shape g.',
    'Mass deficit': 'Mass deficit. (a) Deficit D = 1 &minus; S &minus; P against time in the four runs of a typical M14 '
        'experiment; the shaded last quarter is averaged. (b) Last-quarter deficit against loading for runs 1&ndash;3 '
        'of typical M9, M14 and M16 experiments, with the slope through the origin.',
    'Deficit Km fit': 'Deficit K<sub>m</sub> fit on a typical M1 experiment: deficit per unit of catalyst against '
        'substrate for runs 1&ndash;3, with the fitted D/[cat]<sub>0</sub> = &epsilon;S/(K<sub>m</sub> + S).',
    'Curve shapes': 'Curve shapes. (a) End point of run 1 of a typical M9 experiment. (b) Product gain against time, '
        'both as shares of the run, for run 1 of M9 (fast start, then a stall) and of M7 (slow start, activation). '
        'Dotted lines mark a quarter and half of the run (<code>shape25</code>, <code>shape50</code>) and half of the '
        'gain (<code>t_half_frac</code>); the grey diagonal is a constant rate.',
    'Catalyst loadings': 'Catalyst loadings in 200,000 random training experiments. (a) Loadings of runs 1&ndash;3 '
        '(log scale). (b) Their spread, <code>cat_spread</code>.',
    'Noise and sampling': 'Noise and sampling on a test experiment with 6 points per run and 5% noise (M6). (a) S + P, '
        'which should stay nearly constant; its scatter gives <code>noise_est</code>. (b) Sampling times of each run; '
        'the thick segment is the longest gap. Feature values in the panels are computed on the plotted data.',
}
COMMON_CAPTION = (' Typical experiments are training experiments (20 points per run) closest to their mechanism&rsquo;s '
                  'median; feature values in the panels are computed on the plotted data.')
NO_COMMON = ('Catalyst loadings', 'Noise and sampling')

# first column of a description row -> the panel that shows it
FIG_REF = {
    'vtna_order_all': 'A1c', 'vtna_rms_all': 'A1c', 'vtna_rms_noise_all': 'A1c', 'vtna_width_all': 'A1c',
    'vtna_res_low_all': 'A1c', 'vtna_res_high_all': 'A1c',
    'cat_order_12': 'S2b', 'X_cat_order_12': 'S2b',
    'same_excess_log_ratio': 'S3', 'X_same_excess_log_ratio': 'S3', 'same_excess_mean': 'S3',
    'same_excess_gap_5_20': 'S3',
    'selwyn_12': 'S4', 'selwyn_rms': 'S4',
    'mm_km': 'S5', 'mm_sat': 'S5', 'mm_a': 'S5', 'mm_b': 'S5', 'mm_rmse': 'S5', 'mm_res_1': 'S5', 'mm_res_4': 'S5',
    'law_rel_free': 'S6b', 'law_loss_free': 'S6a', 'law_res4_free': 'S6a', 'law_best': 'S6b',
    'inh_gain_poison': 'S7b', 'inh_amount_poison': 'S7a', 'inh_res4_poison': 'S7a', 'inh_gain_free_0.5': 'S7b',
    'inh_gain_sub_0.5': 'S7b', 'inh_best_free': 'S7b', 'inh_vs_law': 'A6b, A7b',
    'deficit_slope': 'S8b', 'deficit_per_cat_1': 'S8a',
    'dk_km1': 'S9', 'dk_km2': 'S9', 'dk_eps': 'S9', 'dk_rmse': 'S9',
    'P_end_1': 'S10a', 'X_P_end_1': 'S10a', 't_end_1': 'S10a', 'shape25_1': 'S10b', 'shape50_1': 'S10b',
    't_half_frac_1': 'S10b', 'X_shape25_1': 'S10b', 'X_shape50_1': 'S10b', 'X_t_half_frac_1': 'S10b',
    'cat0_1': 'S11a', 'cat_spread': 'S11b',
    'noise_est': 'S12a', 'max_gap_1': 'S12b',
}


def features_cell(cs, kept):
    def fill(p):
        for k, c in enumerate(cs):
            if k:
                p.add_run().add_break()
            r = p.add_run(c if c in kept else c + ' †')
            r.font.name, r.font.size = MONO, Pt(8.5)
            if c not in kept:
                r.font.color.rgb = GREY
    return fill


def build_a():
    all_cols, kept = list(COLUMNS), set(keep_columns('84'))
    fam_all = families(all_cols)
    check(all_cols, fam_all)
    blocks = list(dict.fromkeys(BLOCK_OF[f] for f in C.FAMILY_ORDER))
    n_kept = lambda fs: sum(1 for f in fs for c in fam_all[f] if c in kept)     # noqa: E731
    n_all = lambda fs: sum(len(fam_all[f]) for f in fs)                           # noqa: E731
    fig_no = {f: k + 1 for k, f in enumerate(C.FAMILY_ORDER)}

    doc = Document()
    setup(doc)
    doc.add_paragraph('Supporting information. Engineered features by family', style='Title')
    rich(doc.add_paragraph(), 'Each experiment is four simulated runs of substrate S and product P over time (%s). '
         'The %d features below turn those runs into the numbers chemists read from kinetic tests: catalyst orders, '
         'overlay tests, fitted rate laws, the mass balance, and the context needed to read them. They are grouped '
         'into %d families, and the families into four blocks. The Source column names the paper each '
         'feature&rsquo;s method comes from; &ldquo;This work&rdquo; marks measures designed for this project. Full '
         'references are at the end.' % (cite('bures2023'), len(all_cols), len(C.FAMILY_ORDER)))
    rich(doc.add_paragraph(), 'A redundancy filter on the training data then removed %d of them. Wherever two '
         'features had a Pearson |r| &ge; 0.9, the one with less mutual information with the mechanism was removed '
         '(32 quantile bins, missing values as their own bin), and the correlations were recomputed until no pair was '
         'left at or above 0.9. <b>Features marked &dagger; were removed</b>; the other %d are what the final model '
         'reads.' % (len(all_cols) - len(kept), len(kept)))
    rich(doc.add_paragraph(), 'Each family opens with a reference figure (Figures S1&ndash;S%d) drawn on real '
         'experiments from the dataset, and the Fig. column names the panel that shows each feature.'
         % len(C.FAMILY_ORDER))

    rows, rule = [], []
    for b in blocks:
        for i, f in enumerate([f for f in C.FAMILY_ORDER if BLOCK_OF[f] == b]):
            if i == 0 and rows:
                rule.append(len(rows) + 1)
            rows.append([BLOCK_TITLE[b] if i == 0 else '', FAMILY_NAME[f], '%d' % n_all([f]), '%d' % n_kept([f])])
    rows.append(['<b>Total</b>', '', '<b>%d</b>' % len(all_cols), '<b>%d</b>' % len(kept)])
    rule.append(len(rows))
    caption(doc, 'Table S1.', 'Feature families and the features kept after the redundancy filter.', keep_next=True)
    table(doc, ['Block', 'Family', 'Features', 'Kept after filter'], rows, [2.2, 2.2, 0.9, 1.2],
          align=['l', 'l', 'r', 'r'], size=10, rule_before=rule)

    doc.add_paragraph('Terms used below', style='Heading 2')
    for t in ('<b>Runs 1&ndash;3</b> start with the same substrate, and catalyst loading rises from run 1 to run 3 '
              '(true in 99.998% of training experiments).',
              '<b>Run 4</b> is the same-excess run. It starts with product already present and less substrate, at '
              'the same loading as one of runs 1&ndash;3.',
              '<b><code>X_</code> prefix:</b> computed from pooled conversion X = (1 &minus; S + P)/2 instead of product '
              'alone. Averaging S and P halves the noise variance; using the mass balance to reduce noise follows '
              'reaction-extent reconciliation (%s).' % cite('bonvin2015'),
              '<b>Fitted families</b> fit runs 1&ndash;3 only. Run 4 is held out, so a run-4 residual shows whether '
              'the fitted law predicts a run it never saw. Residuals are scaled by the mean '
              '[cat]<sub>0</sub>&middot;t so experiments can be compared.'):
        rich(doc.add_paragraph(style='List Bullet'), t, 10.5)

    for b in blocks:
        fams = [f for f in C.FAMILY_ORDER if BLOCK_OF[f] == b]
        doc.add_paragraph('%s (%d features, %d kept)' % (BLOCK_TITLE[b], n_all(fams), n_kept(fams)),
                          style='Heading 1')
        for f in fams:
            note, rows_f = DESCRIPTIONS[f]
            doc.add_paragraph('%s · %d features, %d kept' % (FAMILY_NAME[f], n_all([f]), n_kept([f])),
                              style='Heading 2')
            k = fig_no[f]
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_after = Pt(2)
            path = FIGS / ('figS%d.png' % k)
            from PIL import Image
            with Image.open(path) as im:
                w_in = im.size[0] / (im.info.get('dpi', (300, 300))[0] or 300)
            p.add_run().add_picture(str(path), width=Inches(min(w_in, TEXT_W)))
            caption(doc, 'Figure S%d.' % k, deactivation(FIG_CAPTION[f]) + ('' if f in NO_COMMON else COMMON_CAPTION))
            if note:
                p = rich(doc.add_paragraph(), deactivation(note), 10)
                p.paragraph_format.keep_with_next = True
            trows = []
            for cs, meaning, src in rows_f:
                trows.append([features_cell(cs, kept), deactivation(meaning), source_cell(src),
                              FIG_REF.get(cs[0], '')])
            table(doc, ['Feature', 'Meaning', 'Source', 'Fig.'], trows, [1.6, 3.25, 1.15, 0.5])
            doc.add_paragraph().paragraph_format.space_after = Pt(2)

    doc.add_paragraph('References', style='Heading 1')
    for key in sorted(REFS, key=lambda k: REFS[k][1]):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        rich(p, REFS[key][1], 9.5)
    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_A)
    print('wrote', OUT_A)


# ------------------------------------------------------------------------------------------------ appendix B
TABLE_COLUMNS = {   # family -> [(header, function of params)]
    'decision_tree': [('Max depth', lambda p: p['max_depth']), ('Min leaf', lambda p: p['min_samples_leaf'])],
    'random_forest': [('Features per split', lambda p: 'sqrt(p)' if p['max_features'] == 'sqrt'
                       else '%d%%' % round(100 * p['max_features'])),
                      ('Min leaf', lambda p: p['min_samples_leaf'])],
    'hgb': [('Learning rate', lambda p: fmt_num(p['learning_rate'])), ('Leaves', lambda p: p['max_leaf_nodes']),
            ('Min leaf', lambda p: p['min_samples_leaf']), ('L2', lambda p: fmt_num(p['l2_regularization'])),
            ('Features per split', lambda p: '%d%%' % round(100 * p['max_features']))],
    'lightgbm': [('Learning rate', lambda p: fmt_num(p['learning_rate'])), ('Leaves', lambda p: p['num_leaves']),
                 ('Min leaf', lambda p: p['min_child_samples']), ('L2', lambda p: fmt_num(p['reg_lambda'])),
                 ('Features per split', lambda p: '%d%%' % round(100 * p['feature_fraction_bynode'])),
                 ('Rows per tree', lambda p: '%d%%' % round(100 * p['subsample']))],
    'xgboost': [('Learning rate', lambda p: fmt_num(p['learning_rate'])),
                ('Tree size', lambda p: ('%d leaves' % p['max_leaves']) if p.get('grow_policy') == 'lossguide'
                 else 'depth %d' % p['max_depth']),
                ('Min child weight', lambda p: p['min_child_weight']), ('L2', lambda p: fmt_num(p['reg_lambda'])),
                ('Features per split', lambda p: '%d%%' % round(100 * p['colsample_bynode'])),
                ('Rows per tree', lambda p: '%d%%' % round(100 * p['subsample']))],
}
ORDER_B = ['decision_tree', 'random_forest', 'hgb', 'xgboost', 'lightgbm']
TITLE_B = {'decision_tree': 'Decision tree', 'random_forest': 'Random forest (200 trees)',
           'hgb': 'Gradient boosting, scikit-learn (HGB)', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM'}


def build_b():
    s = pd.read_csv(C.TABLES / 'search.csv')
    s['params'] = s.params.map(json.loads)
    w = json.load(open(C.TABLES / 'winner.json'))
    best = {b['model']: b for b in w['model_best']}
    n_sub = C.N_SEARCH
    n_full = C.N_TRAIN_BATCHES * C.BATCH
    doc = Document()
    setup(doc)
    doc.add_paragraph('Appendix 5. Summary of hyperparameter optimisation across the six models', style='Title')
    rich(doc.add_paragraph(), 'Every model was fitted on the same %s training experiments and scored on the '
         'validation set (%s experiments drawn with the paper&rsquo;s sampler: 3&ndash;20 points per run, '
         '0&ndash;2%% noise). The tables give the validation accuracy and log loss (bits) of each setting; the bold '
         'row is each model&rsquo;s best setting. The decision tree and random forest were tuned on small grids. For '
         'each boosting library, setting 0 is the setting found by an earlier search (the same values in all three '
         'libraries where they share a parameter), and settings 1&ndash;7 are random draws (Bergstra &amp; Bengio '
         '2012). Boosting used early stopping on validation accuracy '
         '(patience %d rounds, at most %s); Trees is the number of rounds kept. The best model, LightGBM setting %d, was '
         'then refitted on %s experiments.'
         % (format(n_sub, ','), format(50_000, ','), C.PATIENCE_SEARCH, format(C.MAX_ROUNDS, ','), w['setting'],
            format(n_full, ',')))
    rich(doc.add_paragraph(), 'Min leaf = minimum experiments per leaf (minimum hessian weight per child for '
         'XGBoost); L2 = L2 penalty on leaf values; features per split and rows per tree = fractions sampled; '
         'sqrt(p) = the square root of the number of features.', 10)
    for k, f in enumerate(ORDER_B, start=1):
        g = s[s.model == f].sort_values('setting')
        cols = TABLE_COLUMNS[f]
        boosting = f in ('hgb', 'xgboost', 'lightgbm')
        header = ['Setting'] + [c for c, _ in cols] + (['Trees'] if boosting else []) + ['Val. acc. (%)',
                                                                                          'Log loss']
        rows, bold = [], []
        for i, (_, x) in enumerate(g.iterrows(), start=1):
            vals = ['%d' % x.setting] + ['%s' % fn(x.params) for _, fn in cols]
            if boosting:
                vals.append('%d' % x.rounds if not pd.isna(x.rounds) else '')
            vals += [pct(x.val_accuracy), '%.3f' % x.val_logloss_bits]
            if x.setting == best[f]['setting']:
                bold.append(i)
                vals = ['<b>%s</b>' % v for v in vals]
            rows.append(vals)
        n = len(header)
        first, last = 0.6, [0.65] * ((1 if boosting else 0)) + [0.85, 0.7]
        mid = (TEXT_W - first - sum(last)) / (n - 1 - len(last))
        widths = [first] + [mid] * (n - 1 - len(last)) + last
        caption(doc, 'Table 5%s.' % 'ABCDE'[k - 1], TITLE_B[f] + '.', keep_next=True)
        table(doc, header, rows, widths, align=['l'] + ['r'] * (n - 1), size=9.5)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    rich(doc.add_paragraph(), 'Reference: Bergstra J, Bengio Y (2012). Random search for hyper-parameter '
         'optimization. <i>Journal of Machine Learning Research</i> 13, 281&ndash;305.', 9.5)
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_B)
    print('wrote', OUT_B)


# ------------------------------------------------------------------------------------------------ appendix 6
TITLE_6 = {'Pooled order (VTNA)': 'VTNA features', 'Catalyst order (run pairs)': 'Catalyst order features',
           'Same-excess test': 'Same-excess test features', 'Selwyn test': 'Selwyn test features',
           'Integrated fit (MM)': 'Integrated fit (MM) features', 'Death laws': 'Deactivation laws features',
           'Inhibitor laws': 'Inhibitor laws features', 'Mass deficit': 'Mass deficit features',
           'Deficit Km fit': 'Deficit K<sub>m</sub> fit features', 'Curve shapes': 'Curve shapes features',
           'Catalyst loadings': 'Catalyst loading features', 'Noise and sampling': 'Noise and sampling features'}


def plain(html_text):
    import html
    return html.unescape(re.sub(r'<[^>]+>', '', html_text))


def build_6():
    """Report Appendix 6: all 136 features, one table per family (Tables 6A-6L), features removed by the correlation
    filter marked with a dagger. The same rows are written to a CSV (one row per feature)."""
    all_cols, kept = list(COLUMNS), set(keep_columns('84'))
    check(all_cols, families(all_cols))
    doc = Document()
    setup(doc)
    doc.add_paragraph('Appendix 6. All %d engineered features' % len(all_cols), style='Title')
    rich(doc.add_paragraph(), 'Features marked &dagger; were removed by the correlation filter; the other %d are what '
         'the final model reads.' % len(kept))
    rows_csv = []
    for k, f in enumerate(C.FAMILY_ORDER):
        label = '6' + 'ABCDEFGHIJKL'[k]
        _, rows_f = DESCRIPTIONS[f]
        caption(doc, 'Table %s.' % label, TITLE_6[f] + '.', keep_next=True)
        table(doc, ['Feature', 'Meaning'], [[features_cell(cs, kept), deactivation(m)] for cs, m, _ in rows_f],
              [1.9, 4.6])
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        for cs, m, _ in rows_f:
            for c in cs:
                rows_csv.append({'table': label, 'family': plain(TITLE_6[f])[:-len(' features')], 'feature': c,
                                 'removed': c not in kept, 'meaning': plain(deactivation(m))})
    OUT_6.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_6)
    pd.DataFrame(rows_csv).to_csv(OUT_6.with_suffix('.csv'), index=False)
    print('wrote', OUT_6, 'and', OUT_6.with_suffix('.csv').name)


def main():
    import sys
    todo = sys.argv[1:] or ['5', '6', 'supporting']
    for k in todo:
        {'5': build_b, '6': build_6, 'supporting': build_a}[k]()


if __name__ == '__main__':
    main()
