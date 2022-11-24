import collections

import numpy as np
import pytest
import scipy.special
import scipy.stats

import audmath


@pytest.mark.parametrize(
    'y, expected_x',
    [
        (0, -np.Inf),
        (1, np.Inf),
        ([0, 1], np.array([-np.Inf, np.Inf])),
        (np.array([0, 1]), np.array([-np.Inf, np.Inf])),
    ]
)
def test_ndtri(y, expected_x):
    x = audmath.inverse_normal_distribution(y)
    np.testing.assert_allclose(x, expected_x)
    if isinstance(x, np.ndarray):
        assert np.issubdtype(x.dtype, np.floating)
    else:
        np.issubdtype(type(x), np.floating)


@pytest.mark.parametrize(
    'y',
    [
        0,
        np.exp(-32),
        0.1,
        0.2,
        0.3,
        1,
        -1,
        10,
        np.linspace(0, 1, 50),
    ]
)
def test_scipy_ndtri(y):
    x = audmath.inverse_normal_distribution(y)
    np.testing.assert_allclose(x, scipy.special.ndtri(y))
    np.testing.assert_allclose(x, scipy.stats.norm.ppf(y))


@pytest.mark.parametrize(
    'x, expected',
    [
        ([], np.NaN),
        (0, 0),
        (0.5, 0.5),
        (3, 3),
        ([3], 3),
        (np.array([3]), 3),
        (np.array([[3]]), 3),
        ([0, 1, 2, 3], 1.8708286933869707),
        (np.array([0, 1, 2, 3]), 1.8708286933869707),
    ],
)
def test_rms(x, expected):
    y = audmath.rms(x)
    if np.isnan(expected):
        assert np.isnan(y)
    else:
        assert y == expected


@pytest.mark.parametrize(
    'x, expected',
    [
        ([], -120),
        (0, -120),
        (0.5, -6.020599913279624),
        (3, 9.542425094393248),
        ([3], 9.542425094393248),
        (np.array([3]), 9.542425094393248),
        (np.array([[3]]), 9.542425094393248),
        ([0, 1, 2, 3], 5.440680443502757),
        (np.array([0, 1, 2, 3]), 5.440680443502757),
    ],
)
def test_rms_db(x, expected):
    assert audmath.rms_db(x) == expected
