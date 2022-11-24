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
    'x, axis, keepdims, expected',
    [
        ([], None, False, 0.),
        ([], 0, False, 0.),
        ([], None, True, np.array([0.])),
        ([], 0, True, np.array([0.])),
        (np.array([]), None, False, 0.),
        (np.array([]), 0, False, 0.),
        (np.array([]), None, True, np.array([0.])),
        (np.array([]), 0, True, np.array([0.])),
        (np.array([[]]), None, False, 0.),
        (np.array([[]]), 0, False, 0.),
        (np.array([[]]), 1, False, 0.),
        (np.array([[]]), None, True, np.array([[0.]])),
        (np.array([[]]), 0, True, np.array([[0.]])),
        (np.array([[]]), 1, True, np.array([[0.]])),
        (0, None, False, 0.),
        (0.5, None, False, 0.5),
        (3, None, False, 3.),
        ([3], None, False, 3.),
        ([3], 0, False, 3.),
        ([3], None, True, np.array([3.])),
        ([3], 0, True, np.array([3.])),
        (np.array([3]), None, False, 3.),
        (np.array([3]), 0, False, 3.),
        (np.array([3]), None, True, np.array([3.])),
        (np.array([3]), 0, True, np.array([3.])),
        (np.array([[3]]), None, False, 3.),
        (np.array([[3]]), 0, False, 3.),
        (np.array([[3]]), None, True, np.array([[3.]])),
        (np.array([[3]]), 0, True, np.array([[3.]])),
        ([0, 1, 2, 3], None, False, 1.8708286933869707),
        ([0, 1, 2, 3], 0, False, 1.8708286933869707),
        ([0, 1, 2, 3], None, True, np.array([1.8708286933869707])),
        ([0, 1, 2, 3], 0, True, np.array([1.8708286933869707])),
        (np.array([0, 1, 2, 3]), None, False, 1.8708286933869707),
        (np.array([0, 1, 2, 3]), 0, False, 1.8708286933869707),
        (np.array([0, 1, 2, 3]), None, True, np.array([1.8708286933869707])),
        (np.array([0, 1, 2, 3]), 0, True, np.array([1.8708286933869707])),
        (
            [[0, 1], [2, 3]],
            None,
            False,
            1.8708286933869707,
        ),
        (
            [[0, 1], [2, 3]],
            0,
            False,
            np.array([1.4142135623730951, 2.23606797749979]),
        ),
        (
            [[0, 1], [2, 3]],
            1,
            False,
            np.array([0.7071067811865476, 2.5495097567963922]),
        ),
        (
            [[0, 1], [2, 3]],
            None,
            True,
            np.array([[1.8708286933869707]]),
        ),
        (
            [[0, 1], [2, 3]],
            0,
            True,
            np.array([[1.4142135623730951], [2.23606797749979]]).T,
        ),
        (
            [[0, 1], [2, 3]],
            1,
            True,
            np.array([[0.7071067811865476], [2.5495097567963922]]),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            False,
            3.,
            marks=pytest.mark.xfail(raises=np.AxisError),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            True,
            3.,
            marks=pytest.mark.xfail(raises=np.AxisError),
        ),
    ],
)
def test_rms(x, axis, keepdims, expected):
    y = audmath.rms(x, axis=axis, keepdims=keepdims)
    np.testing.assert_array_equal(y, expected)


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
