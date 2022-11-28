import collections

import numpy as np
import pytest
import scipy.special
import scipy.stats

import audmath


@pytest.mark.parametrize(
    'x, bottom, expected_y',
    [
        (0, None, -np.Inf),
        (0, -120, -120),
        (-1, None, -np.Inf),
        (-1, -120, -120),
        (0., None, -np.Inf),
        (0., -120, -120),
        (-1., None, -np.Inf),
        (-1., -120, -120),
        ([], None, np.array([])),
        ([], -120, np.array([])),
        (np.array([]), None, np.array([])),
        (np.array([]), -120, np.array([])),
        ([[]], None, np.array([[]])),
        ([[]], -120, np.array([[]])),
        (np.array([[]]), None, np.array([[]])),
        (np.array([[]]), -120, np.array([[]])),
        ([0, 1], None, np.array([-np.Inf, 0.])),
        ([0, 1], -120, np.array([-120, 0.])),
        ([0., 1.], None, np.array([-np.Inf, 0.])),
        ([0., 1.], -120, np.array([-120, 0.])),
        (np.array([0, 1]), None, np.array([-np.Inf, 0.])),
        (np.array([0, 1]), -120, np.array([-120, 0.])),
        (np.array([0., 1.]), None, np.array([-np.Inf, 0.])),
        (np.array([0., 1.]), -120, np.array([-120, 0.])),
        (np.array([[0], [1]]), None, np.array([[-np.Inf], [0.]])),
        (np.array([[0], [1]]), -120, np.array([[-120], [0.]])),
        (np.array([[0.], [1.]]), None, np.array([[-np.Inf], [0.]])),
        (np.array([[0.], [1.]]), -120, np.array([[-120], [0.]])),
    ],
)
def test_db(x, bottom, expected_y):
    y = audmath.db(x, bottom=bottom)
    np.testing.assert_allclose(y, expected_y)
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        np.issubdtype(type(y), np.floating)


@pytest.mark.parametrize(
    'shape',
    [
        'linear',
        'kaiser',
        'tukey',
        'exponential',
        'logarithmic',
    ],
)
@pytest.mark.parametrize(
    'samples, level, expected',
    [
        (-1, -120, np.array([])),
        (0, -120, np.array([])),
        (1, -120, np.array([0])),
        (2, -120, np.array([0, 1])),
        (1, -20, np.array([0.1])),
        (2, -20, np.array([0.1, 1])),
    ]
)
def test_fadein_level(shape, samples, level, expected):
    win = audmath.fadein(samples, shape=shape, level=level)
    np.testing.assert_allclose(win, expected)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    'samples, shape, expected',
    [
        (3, 'linear', np.array([0, 0.5, 1])),
        (3, 'kaiser', np.array([0, 4.6272e-01, 1])),
        (3, 'tukey', np.array([0, 0.5, 1])),
        (3, 'exponential', np.array([0, 0.26894142, 1])),
        (3, 'logarithmic', np.array([0, 0.63092975, 1])),
    ]
)
def test_fadein_shape(samples, shape, expected):
    win = audmath.fadein(samples, shape=shape)
    np.testing.assert_allclose(win, expected, rtol=1e-05)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    'shape, error, error_msg',
    [
        (
            'unknown',
            ValueError,
            (
                "shape has to be one of the following: "
                f"{(', ').join(audmath.core.api.FADEIN_SHAPES)},"
                f"not 'unknown'."
            ),
        ),
    ],
)
def test_fadein_error(shape, error, error_msg):
    with pytest.raises(error, match=error_msg):
        audmath.fadein(3, shape=shape)


@pytest.mark.parametrize(
    'y, bottom, expected_x',
    [
        (0, None, 1.),
        (0, -120, 1.),
        (0., None, 1.),
        (0., -120, 1.),
        (-np.Inf, None, 0.),
        (-np.Inf, -120, 0.),
        (-160, None, 1e-08),
        (-160, -120, 0.),
        (-160., None, 1e-08),
        (-160., -120, 0.),
        (-120, None, 1e-06),
        (-120, -120, 0.),
        (-120., None, 1e-06),
        (-120., -120, 0.),
        (-1, None, 0.8912509381337456),
        (-1, -120, 0.8912509381337456),
        (-1., None, 0.8912509381337456),
        (-1., -120, 0.8912509381337456),
        ([-np.Inf, -120], None, np.array([0., 1e-06])),
        ([-np.Inf, -120], -120, np.array([0., 0.])),
        ([], None, np.array([])),
        ([], -120, np.array([])),
        (np.array([]), None, np.array([])),
        (np.array([]), -120, np.array([])),
        ([[]], None, np.array([[]])),
        ([[]], -120, np.array([[]])),
        (np.array([[]]), None, np.array([[]])),
        (np.array([[]]), -120, np.array([[]])),
        ([0, -1], None, np.array([1., 0.8912509381337456])),
        ([0, -1], -120, np.array([1., 0.8912509381337456])),
        ([0., -1.], None, np.array([1., 0.8912509381337456])),
        ([0., -1.], -120, np.array([1., 0.8912509381337456])),
        (np.array([-np.Inf, -120]), None, np.array([0., 1e-06])),
        (np.array([-np.Inf, -120]), -120, np.array([0., 0.])),
        (np.array([0, -1]), None, np.array([1., 0.8912509381337456])),
        (np.array([0, -1]), -120, np.array([1., 0.8912509381337456])),
        (np.array([0., -1.]), None, np.array([1., 0.8912509381337456])),
        (np.array([0., -1.]), -120, np.array([1., 0.8912509381337456])),
        (np.array([[-np.Inf], [-120]]), None, np.array([[0.], [1e-06]])),
        (np.array([[-np.Inf], [-120]]), -120, np.array([[0.], [0.]])),
        (np.array([[0], [-1]]), None, np.array([[1.], [0.8912509381337456]])),
        (np.array([[0], [-1]]), -120, np.array([[1.], [0.8912509381337456]])),
        (
            np.array([[0.], [-1.]]),
            None,
            np.array([[1.], [0.8912509381337456]]),
        ),
        (
            np.array([[0.], [-1.]]),
            -120,
            np.array([[1.], [0.8912509381337456]]),
        ),
    ],
)
def test_inverse_db(y, bottom, expected_x):
    x = audmath.inverse_db(y, bottom=bottom)
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
