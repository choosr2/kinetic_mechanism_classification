"""Settings shared by every step: folders, random seeds, sample sizes and the decision rules of the paper.

Every number the Methods section quotes is set here once, so the steps cannot disagree with each other.
Folders can be moved with environment variables (useful when the 10 GB dataset already lives elsewhere):
    KMC_RAW_DIR        the Burés & Larrosa pickles          (default: data/raw)
    KMC_PROCESSED_DIR  computed feature matrices            (default: data/processed)
    KMC_MODELS_DIR     fitted models                        (default: models)
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------------------------------- folders
REPO = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get('KMC_RAW_DIR', REPO / 'data' / 'raw'))
PROCESSED = Path(os.environ.get('KMC_PROCESSED_DIR', REPO / 'data' / 'processed'))
MODELS = Path(os.environ.get('KMC_MODELS_DIR', REPO / 'models'))
TABLES = REPO / 'results' / 'tables'            # every computed table (the inputs of the figures and appendices)
REPORT = REPO / 'results' / 'report'            # every item of the report, named as in the report
SUPPORTING = REPO / 'results' / 'supporting'    # figures and a document that are not in the report

# ----------------------------------------------------------------------------------------------------- data
MECHS = ['M%d' % i for i in range(1, 21)]
CATEGORIES = [('Core', 0, 1), ('Bicatalytic', 1, 5), ('Activation', 5, 8), ('Deactivation', 8, 20)]   # [start, end)
CONC_COLS = [1, 2, 4, 5, 7, 8, 10, 11]      # S and P columns of the four runs in each 21 x 12 table

# The authors' sampler (Burés & Larrosa 2023, train.py): every batch keeps one random number of time points
# (t = 0 always kept) and gets one random Gaussian noise level, in % of the starting substrate.
TPS = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
NOISE = [0, 0.5, 1, 2]
BATCH = 2048                                 # training batch: 4,950,000 // 2,048 = 2,416 batches
N_TRAIN_BATCHES = 4_950_000 // BATCH         # the last 2,032 samples do not fill a batch and are left out
TRAIN_RNG_SEED = 1                           # draws of points and noise levels (the authors' seed)
TRAIN_NOISE_SEED = 2                         # the Gaussian noise blocks (the authors' were unseeded)
VAL_BATCH = 25                               # validation batch
VAL_SEED = 1

# test sets: (points per run, noise %), as released by the authors; the main one is listed first
TEST_CONDITIONS = [(6, 0), (6, 1), (6, 5), (20, 1), (2, 1)]
MAIN_TEST = 'test_6pts_0pct'

# ----------------------------------------------------------------------------------------------------- features
# The 12 feature families, by column-name prefix, and the 3 blocks of related families
FAMILY_RULES = [
    ('Curve shapes', ('P_end_', 't_end_', 'shape25_', 'shape50_', 't_half_frac_',
                      'X_P_end_', 'X_shape25_', 'X_shape50_', 'X_t_half_frac_')),
    ('Catalyst loadings', ('cat0_', 'cat_spread')),
    ('Catalyst order (run pairs)', ('cat_order_', 'X_cat_order_')),
    ('Same-excess test', ('same_excess', 'X_same_excess')),
    ('Selwyn test', ('selwyn_',)),
    ('Pooled order (VTNA)', ('vtna_',)),
    ('Mass deficit', ('deficit_',)),
    ('Deficit Km fit', ('dk_',)),
    ('Noise and sampling', ('noise_est', 'max_gap_')),
    ('Integrated fit (MM)', ('mm_',)),
    ('Death laws', ('law_',)),                # called "Deactivation laws" in the paper
    ('Inhibitor laws', ('inh_',)),
]
# order used in tables and figures: graphical tests, fitted laws, mass balance, context
FAMILY_ORDER = ['Pooled order (VTNA)', 'Catalyst order (run pairs)', 'Same-excess test', 'Selwyn test',
                'Integrated fit (MM)', 'Death laws', 'Inhibitor laws', 'Mass deficit', 'Deficit Km fit',
                'Curve shapes', 'Catalyst loadings', 'Noise and sampling']
BLOCKS = {'Mass balance': ['Mass deficit', 'Deficit Km fit'],
          'Fitted laws': ['Integrated fit (MM)', 'Death laws', 'Inhibitor laws'],
          'Graphical tests': ['Catalyst order (run pairs)', 'Same-excess test', 'Selwyn test', 'Pooled order (VTNA)']}

# ----------------------------------------------------------------------------------------------------- EDA
EDA_N = 200_000          # random training experiments used for the feature EDA and the correlation filter
EDA_SEED = 2026
MI_BINS = 32             # quantile bins for mutual information (missing values get their own bin)
CORR_CUT = 0.9           # Pearson |r| at or above which one feature of the pair is removed
FENCE = 3.0              # extreme value: more than 3 IQR beyond the quartiles

# ----------------------------------------------------------------------------------------------------- models
SEED = 1
N_SEARCH = 204_800                 # training rows shared by every model in the search
PATIENCE_SEARCH, PATIENCE_FINAL, MAX_ROUNDS = 50, 100, 3000   # boosting: early stopping on validation accuracy
TIE_POINTS = 0.3                   # models within 0.3 points of the best validation accuracy count as tied
N_JOBS = os.cpu_count()

# Setting 0 of each boosting library: the best setting of an earlier, separate search (scikit-learn gradient
# boosting on an earlier feature set). It is tried alongside 7 random draws per library; it did not win.
SETTING_0 = {'learning_rate': 0.07826859654953973, 'max_leaf_nodes': 35, 'min_samples_leaf': 20,
             'l2_regularization': 4.042074809342264, 'max_features': 0.5}

# ----------------------------------------------------------------------------------------------------- permutation
PERM_SEED = 31                     # the permutation of unit u, repeat r uses default_rng([31, k(u), r])
N_REPEATS = 3
MIN_RECALL, MIN_CELL, N_SE = 2.0, 1.0, 3.0   # significance: >= 2 points (recall) / 1 point (other cells), > 3 SE
AFFECTS = 5.0                      # a family affects a mechanism if it lowers its recall significantly by >= 5 points
NOISY_TESTS = ['test_6pts_1pct', 'test_6pts_5pct']   # the noise family is also permuted on these
# roles of a unit, from L = its significant recall losses (points) over the 20 mechanisms:
#   breadth = (sum L)^2 / sum L^2 (effective number of mechanisms sharing the loss), peak = max L
#   Minor: peak < 5; Group specialist: breadth >= 3 and >= 80% of the loss in one mechanism category;
#   Generalist: breadth >= 6; Specialist: breadth < 3 and peak >= 10; Intermediate: anything else
PEAK_MINOR, PEAK_SPECIALIST, BREADTH_SPECIALIST, BREADTH_GENERALIST, GROUP_SHARE = 5.0, 10.0, 3.0, 6.0, 0.8
