"""The feature code must reproduce the features used in the paper.

tests/data/reference_features.npz holds 40 experiments of the released test set (6 points per run, 1% noise) and
their 136 features as computed for the paper. Run with:  uv run pytest
"""
from pathlib import Path

import numpy as np

from kinetic_mech.features import COLUMNS, compute_features, families, keep_columns

REF = Path(__file__).parent / 'data' / 'reference_features.npz'


def test_features_match_the_paper():
    ref = np.load(REF)
    F = compute_features(ref['x1'], ref['x2']).to_numpy()
    expected = ref['features']
    assert F.shape == expected.shape == (40, 136)
    assert np.array_equal(np.isnan(F), np.isnan(expected))
    ok = ~np.isnan(F)
    np.testing.assert_allclose(F[ok], expected[ok], rtol=1e-9, atol=1e-12)


def test_families_cover_every_feature_once():
    fam = families(COLUMNS)
    assert len(fam) == 12
    assert sorted(c for cs in fam.values() for c in cs) == sorted(COLUMNS)


def test_correlation_filter_keeps_84():
    assert len(keep_columns('84')) == 84
    assert len(keep_columns('136')) == 136
