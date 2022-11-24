import collections

import numpy as np
import pytest
import scipy.special
import scipy.stats

import audmath


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (0, -np.Inf),
        (-1, -np.Inf),
        ([0, 1], np.array([-np.Inf, 0.])),
        (np.array([0, 1]), np.array([-np.Inf, 0.])),
        (np.array([[0], [1]]), np.array([[-np.Inf], [0.]])),
    ],
)
def test_db(x, expected_y):
    y = audmath.db(x)
    np.testing.assert_allclose(y, expected_y)
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        np.issubdtype(type(y), np.floating)


@pytest.mark.parametrize(
    'y, expected_x',
    [
        (0, 1.),
        (-1, 0.8912509381337456),
        ([0, 1], np.array([1., 0.8912509381337456])),
        (np.array([0, 1]), np.array([1., 0.8912509381337456])),
        (np.array([[0], [1]]), np.array([[1.], [0.8912509381337456]])),
    ],
)
def test_inverse_db(y, expected_x):
    x = audmath.inverse_db(y)
    np.testing.assert_allclose(x, expected_x)
    if isinstance(x, np.ndarray):
        assert np.issubdtype(x.dtype, np.floating)
    else:
        np.issubdtype(type(x), np.floating)


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
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        assert np.issubdtype(type(y), np.floating)


@pytest.mark.parametrize(
    'x, axis, keepdims, expected',
    [
        ([], None, False, -120.),
        ([], 0, False, -120.),
        ([], None, True, np.array([-120.])),
        ([], 0, True, np.array([-120.])),
        (np.array([]), None, False, -120.),
        (np.array([]), 0, False, -120.),
        (np.array([]), None, True, np.array([-120.])),
        (np.array([]), 0, True, np.array([-120.])),
        (np.array([[]]), None, False, -120.),
        (np.array([[]]), 0, False, -120.),
        (np.array([[]]), 1, False, -120.),
        (np.array([[]]), None, True, np.array([[-120.]])),
        (np.array([[]]), 0, True, np.array([[-120.]])),
        (np.array([[]]), 1, True, np.array([[-120.]])),
        (0, None, False, -120.),
        (0.5, None, False, -6.020599913279624),
        (3, None, False, 9.542425094393248),
        ([3], None, False, 9.542425094393248),
        ([3], 0, False, 9.542425094393248),
        ([3], None, True, np.array([9.542425094393248])),
        ([3], 0, True, np.array([9.542425094393248])),
        (np.array([3]), None, False, 9.542425094393248),
        (np.array([3]), 0, False, 9.542425094393248),
        (np.array([3]), None, True, np.array([9.542425094393248])),
        (np.array([3]), 0, True, np.array([9.542425094393248])),
        (np.array([[3]]), None, False, 9.542425094393248),
        (np.array([[3]]), 0, False, 9.542425094393248),
        (np.array([[3]]), None, True, np.array([[9.542425094393248]])),
        (np.array([[3]]), 0, True, np.array([[9.542425094393248]])),
        ([0, 1, 2, 3], None, False, 5.440680443502757),
        ([0, 1, 2, 3], 0, False, 5.440680443502757),
        ([0, 1, 2, 3], None, True, np.array([5.440680443502757])),
        ([0, 1, 2, 3], 0, True, np.array([5.440680443502757])),
        (np.array([0, 1, 2, 3]), None, False, 5.440680443502757),
        (np.array([0, 1, 2, 3]), 0, False, 5.440680443502757),
        (np.array([0, 1, 2, 3]), None, True, np.array([5.440680443502757])),
        (np.array([0, 1, 2, 3]), 0, True, np.array([5.440680443502757])),
        (
            [[0, 1], [2, 3]],
            None,
            False,
            5.440680443502757,
        ),
        (
            [[0, 1], [2, 3]],
            0,
            False,
            np.array([3.010299956639812, 6.989700043360188]),
        ),
        (
            [[0, 1], [2, 3]],
            1,
            False,
            np.array([-3.0102999566398116, 8.129133566428555]),
        ),
        (
            [[0, 1], [2, 3]],
            None,
            True,
            np.array([[5.440680443502757]]),
        ),
        (
            [[0, 1], [2, 3]],
            0,
            True,
            np.array([[3.010299956639812], [6.989700043360188]]).T,
        ),
        (
            [[0, 1], [2, 3]],
            1,
            True,
            np.array([[-3.0102999566398116], [8.129133566428555]]),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            False,
            9.542425094393248,
            marks=pytest.mark.xfail(raises=np.AxisError),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            True,
            9.542425094393248,
            marks=pytest.mark.xfail(raises=np.AxisError),
        ),
    ],
)
def test_rms_db(x, axis, keepdims, expected):
    y = audmath.rms_db(x, axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(y, expected)
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        assert np.issubdtype(type(y), np.floating)
