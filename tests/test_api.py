import re

import numpy as np
import pandas as pd
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


week = np.timedelta64(24 * 7, 'h') / np.timedelta64(1, 's')
day = np.timedelta64(24, 'h') / np.timedelta64(1, 's')
hour = np.timedelta64(1, 'h') / np.timedelta64(1, 's')
minute = np.timedelta64(1, 'm') / np.timedelta64(1, 's')
second = np.timedelta64(1, 's') / np.timedelta64(1, 's')
millisecond = np.timedelta64(1, 'ms') / np.timedelta64(1, 's')
microsecond = np.timedelta64(1, 'us') / np.timedelta64(1, 's')
nanosecond = np.timedelta64(1, 'ns') / np.timedelta64(1, 's')


@pytest.mark.parametrize(
    'duration, sampling_rate, expected',
    [
        (None, None, np.NaN),
        (None, 1000, np.NaN),
        ('', None, np.NaN),
        ('', 1000, np.NaN),
        ('none', None, np.NaN),
        ('none', 1000, np.NaN),
        ('None', None, np.NaN),
        ('None', 1000, np.NaN),
        ('nan', None, np.NaN),
        ('nan', 1000, np.NaN),
        ('NaN', None, np.NaN),
        ('NaN', 1000, np.NaN),
        ('nat', None, np.NaN),
        ('nat', 1000, np.NaN),
        ('NaT', None, np.NaN),
        ('NaT', 1000, np.NaN),
        (np.NaN, None, np.NaN),
        (np.NaN, 1000, np.NaN),
        (pd.NaT, None, np.NaN),
        (pd.NaT, 1000, np.NaN),
        (np.timedelta64('NaT', 's'), None, np.NaN),
        (np.timedelta64('NaT', 's'), 1000, np.NaN),
        ('inf', None, np.inf),
        ('inf', 1000, np.inf),
        ('Inf', None, np.inf),
        ('Inf', 1000, np.inf),
        (np.inf, None, np.inf),
        (np.inf, 1000, np.inf),
        (np.Inf, None, np.inf),
        (np.Inf, 1000, np.inf),
        (2, None, 2.0),
        (2, 1000, 0.002),
        (2.0, None, 2.0),
        (2.0, 1000, 0.002),
        ('s', None, 1.0),
        ('s', 1000, 1.0),
        (' s', None, 1.0),
        (' s', 1000, 1.0),
        ('2s', None, 2.0),
        ('2s', 1000, 2.0),
        ('2s ', None, 2.0),
        ('2s ', 1000, 2.0),
        (' 2s', None, 2.0),
        (' 2s', 1000, 2.0),
        ('2 s', None, 2.0),
        ('2 s', 1000, 2.0),
        ('2  s', None, 2.0),
        ('2  s', 1000, 2.0),
        ('2000ms', None, 2.0),
        ('2000ms', 1000, 2.0),
        ('2000.0ms', None, 2.0),
        ('2000.0ms', 1000, 2.0),
        ('2000 ms', None, 2.0),
        ('2000 ms', 1000, 2.0),
        ('2000.0 ms', None, 2.0),
        ('2000.0 ms', 1000, 2.0),
        ('2000', None, 2000.0),
        ('2000', 1000, 2.0),
        ('2000 ', None, 2000.0),
        ('2000 ', 1000, 2.0),
        ('2000.0', None, 2000.0),
        ('2000.0', 1000, 2.0),
        ('2000.1', None, 2000.1),
        ('2000.1', 1000, 2.0000999999999998),
        ('0.5', None, 0.5),
        ('0.5', 2, 0.25),
        ('3', 1.5, 2.0),
        (np.timedelta64(2, 's'), None, 2.0),
        (np.timedelta64(2, 's'), 1000, 2.0),
        (np.timedelta64(2000, 'ms'), None, 2.0),
        (np.timedelta64(2000, 'ms'), 1000, 2.0),
        (pd.to_timedelta(2, 's'), None, 2.0),
        (pd.to_timedelta(2, 's'), 1000, 2.0),
        (pd.to_timedelta(2000, 'ms'), None, 2.0),
        (pd.to_timedelta(2000, 'ms'), 1000, 2.0),
        ('+inf', None, np.inf),
        ('+inf', 1000, np.inf),
        ('+Inf', None, np.inf),
        ('+Inf', 1000, np.inf),
        (+2, None, 2.0),
        (+2, 1000, 0.002),
        (+2.0, None, 2.0),
        (+2.0, 1000, 0.002),
        ('+s', None, 1.0),
        ('+s', 1000, 1.0),
        (' +s', None, 1.0),
        (' +s', 1000, 1.0),
        ('+2s', None, 2.0),
        ('+2s', 1000, 2.0),
        ('+2s ', None, 2.0),
        ('+2s ', 1000, 2.0),
        (' +2s', None, 2.0),
        (' +2s', 1000, 2.0),
        ('+2 s', None, 2.0),
        ('+2 s', 1000, 2.0),
        ('+2  s', None, 2.0),
        ('+2  s', 1000, 2.0),
        ('+2000ms', None, 2.0),
        ('+2000ms', 1000, 2.0),
        ('+2000.0ms', None, 2.0),
        ('+2000.0ms', 1000, 2.0),
        ('+2000 ms', None, 2.0),
        ('+2000 ms', 1000, 2.0),
        ('+2000.0 ms', None, 2.0),
        ('+2000.0 ms', 1000, 2.0),
        ('+2000', None, 2000.0),
        ('+2000', 1000, 2.0),
        ('+2000 ', None, 2000.0),
        ('+2000 ', 1000, 2.0),
        ('+2000.0', None, 2000.0),
        ('+2000.0', 1000, 2.0),
        ('+2000.1', None, 2000.1),
        ('+2000.1', 1000, 2.0000999999999998),
        ('+0.5', None, 0.5),
        ('+0.5', 2, 0.25),
        ('+3', 1.5, 2.0),
        (np.timedelta64(+2, 's'), None, 2.0),
        (np.timedelta64(+2, 's'), 1000, 2.0),
        (np.timedelta64(+2000, 'ms'), None, 2.0),
        (np.timedelta64(+2000, 'ms'), 1000, 2.0),
        (pd.to_timedelta(+2, 's'), None, 2.0),
        (pd.to_timedelta(+2, 's'), 1000, 2.0),
        (pd.to_timedelta(+2000, 'ms'), None, 2.0),
        (pd.to_timedelta(+2000, 'ms'), 1000, 2.0),
        ('-inf', None, -np.inf),
        ('-inf', 1000, -np.inf),
        ('-Inf', None, -np.inf),
        ('-Inf', 1000, -np.inf),
        (-2, None, -2.0),
        (-2, 1000, -0.002),
        (-2.0, None, -2.0),
        (-2.0, 1000, -0.002),
        ('-s', None, -1.0),
        ('-s', 1000, -1.0),
        (' -s', None, -1.0),
        (' -s', 1000, -1.0),
        ('-2s', None, -2.0),
        ('-2s', 1000, -2.0),
        ('-2s ', None, -2.0),
        ('-2s ', 1000, -2.0),
        (' -2s', None, -2.0),
        (' -2s', 1000, -2.0),
        ('-2 s', None, -2.0),
        ('-2 s', 1000, -2.0),
        ('-2  s', None, -2.0),
        ('-2  s', 1000, -2.0),
        ('-2000ms', None, -2.0),
        ('-2000ms', 1000, -2.0),
        ('-2000.0ms', None, -2.0),
        ('-2000.0ms', 1000, -2.0),
        ('-2000 ms', None, -2.0),
        ('-2000 ms', 1000, -2.0),
        ('-2000.0 ms', None, -2.0),
        ('-2000.0 ms', 1000, -2.0),
        ('-2000', None, -2000.0),
        ('-2000', 1000, -2.0),
        ('-2000 ', None, -2000.0),
        ('-2000 ', 1000, -2.0),
        ('-2000.0', None, -2000.0),
        ('-2000.0', 1000, -2.0),
        ('-2000.1', None, -2000.1),
        ('-2000.1', 1000, -2.0000999999999998),
        ('-0.5', None, -0.5),
        ('-0.5', 2, -0.25),
        ('-3', 1.5, -2.0),
        (np.timedelta64(-2, 's'), None, -2.0),
        (np.timedelta64(-2, 's'), 1000, -2.0),
        (np.timedelta64(-2000, 'ms'), None, -2.0),
        (np.timedelta64(-2000, 'ms'), 1000, -2.0),
        (pd.to_timedelta(-2, 's'), None, -2.0),
        (pd.to_timedelta(-2, 's'), 1000, -2.0),
        (pd.to_timedelta(-2000, 'ms'), None, -2.0),
        (pd.to_timedelta(-2000, 'ms'), 1000, -2.0),
        # week
        ('1W', None, week),
        # day
        ('1D', None, day),
        ('1days', None, day),
        ('1day', None, day),
        # hour
        ('1h', None, hour),
        ('1hours', None, hour),
        ('1hour', None, hour),
        ('1hr', None, hour),
        # minute
        ('1m', None, minute),
        ('1minutes', None, minute),
        ('1minute', None, minute),
        ('1min', None, minute),
        ('1T', None, minute),
        # second
        ('1s', None, second),
        ('1seconds', None, second),
        ('1second', None, second),
        ('1sec', None, second),
        ('1S', None, second),
        # millisecond
        ('1ms', None, millisecond),
        ('1milliseconds', None, millisecond),
        ('1millisecond', None, millisecond),
        ('1millis', None, millisecond),
        ('1milli', None, millisecond),
        ('1L', None, millisecond),
        # microsecond
        ('1us', None, microsecond),
        ('1μs', None, microsecond),
        ('1microseconds', None, microsecond),
        ('1microsecond', None, microsecond),
        ('1micros', None, microsecond),
        ('1micro', None, microsecond),
        ('1U', None, microsecond),
        # nanosecond
        ('1ns', None, nanosecond),
        ('1nanoseconds', None, nanosecond),
        ('1nanosecond', None, nanosecond),
        ('1nanos', None, nanosecond),
        ('1nano', None, nanosecond),
        ('1N', None, nanosecond),
    ]
)
def test_duration_in_seconds(duration, sampling_rate, expected):
    duration_in_seconds = audmath.duration_in_seconds(duration, sampling_rate)
    if np.isnan(expected):
        assert np.isnan(duration_in_seconds)
    else:
        assert duration_in_seconds == expected


@pytest.mark.parametrize(
    'duration, sampling_rate, error, error_msg',
    [
        (
            '2abc',
            None,
            ValueError,
            "The provided unit 'abc' is not known.",
        ),
        (
            '2 abc',
            None,
            ValueError,
            "The provided unit 'abc' is not known.",
        ),
        (
            '2a bc',
            None,
            ValueError,
            (
                "Your given duration '2a bc' "
                "is not conform to the <value><unit> pattern."
            )
        ),
        (
            '2.0a bc',
            None,
            ValueError,
            (
                "Your given duration '2.0a bc' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            ' ',
            None,
            ValueError,
            (
                "Your given duration ' ' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            '  ',
            None,
            ValueError,
            (
                "Your given duration '  ' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            '1 0 ms',
            None,
            ValueError,
            (
                "Your given duration '1 0 ms' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            '10 m s',
            None,
            ValueError,
            (
                "Your given duration '10 m s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            '1 0 m s',
            None,
            ValueError,
            (
                "Your given duration '1 0 m s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            '2.m5s',
            None,
            ValueError,
            (
                "Your given duration '2.m5s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
    ]
)
def test_duration_in_seconds_error(duration, sampling_rate, error, error_msg):
    with pytest.raises(error, match=re.escape(error_msg)):
        audmath.duration_in_seconds(duration, sampling_rate)


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
    'samples, half, expected',
    [
        (-1, 'left', np.array([])),
        (0, 'left', np.array([])),
        (1, 'left', np.array([0])),
        (2, 'left', np.array([0, 1])),
        (-1, 'right', np.array([])),
        (0, 'right', np.array([])),
        (1, 'right', np.array([0])),
        (2, 'right', np.array([1, 0])),
        (-1, None, np.array([])),
        (0, None, np.array([])),
        (1, None, np.array([0])),
        (2, None, np.array([0, 0])),
        (3, None, np.array([0, 1, 0])),
    ]
)
def test_window_level(shape, samples, half, expected):
    win = audmath.window(samples, shape=shape, half=half)
    np.testing.assert_allclose(win, expected)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    'samples, shape, half, expected',
    [
        (3, 'linear', 'left', np.array([0, 0.5, 1])),
        (3, 'kaiser', 'left', np.array([0, 4.6272e-01, 1])),
        (3, 'tukey', 'left', np.array([0, 0.5, 1])),
        (3, 'exponential', 'left', np.array([0, 0.26894142, 1])),
        (3, 'logarithmic', 'left', np.array([0, 0.63092975, 1])),
        (3, 'linear', 'right', np.array([1, 0.5, 0])),
        (3, 'kaiser', 'right', np.array([1, 4.6272e-01, 0])),
        (3, 'tukey', 'right', np.array([1, 0.5, 0])),
        (3, 'exponential', 'right', np.array([1, 0.26894142, 0])),
        (3, 'logarithmic', 'right', np.array([1, 0.63092975, 0])),
        (5, 'linear', None, np.array([0, 0.5, 1, 0.5, 0])),
        (5, 'kaiser', None, np.array([0, 4.6272e-01, 1, 4.6272e-01, 0])),
        (5, 'tukey', None, np.array([0, 0.5, 1, 0.5, 0])),
        (5, 'exponential', None, np.array([0, 0.26894142, 1, 0.26894142, 0])),
        (5, 'logarithmic', None, np.array([0, 0.63092975, 1, 0.63092975, 0])),
        (4, 'linear', None, np.array([0, 0.5, 0.5, 0])),
        (4, 'kaiser', None, np.array([0, 4.6272e-01, 4.6272e-01, 0])),
        (4, 'tukey', None, np.array([0, 0.5, 0.5, 0])),
        (4, 'exponential', None, np.array([0, 0.26894142, 0.26894142, 0])),
        (4, 'logarithmic', None, np.array([0, 0.63092975, 0.63092975, 0])),
    ]
)
def test_window_shape(samples, shape, half, expected):
    win = audmath.window(samples, shape=shape, half=half)
    np.testing.assert_allclose(win, expected, rtol=1e-05)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    'shape, half, error, error_msg',
    [
        (
            'unknown',
            None,
            ValueError,
            (
                "shape has to be one of the following: "
                f"{(', ').join(audmath.core.api.WINDOW_SHAPES)},"
                f"not 'unknown'."
            ),
        ),
        (
            'linear',
            'center',
            ValueError,
            (
                "half has to be 'left' or 'right' "
                "not 'center'."
            ),
        ),
    ],
)
def test_window_error(shape, half, error, error_msg):
    with pytest.raises(error, match=error_msg):
        audmath.window(3, shape=shape, half=half)
