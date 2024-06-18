import re

import numpy as np
import pandas as pd
import pytest

import audmath


week = np.timedelta64(24 * 7, "h") / np.timedelta64(1, "s")
day = np.timedelta64(24, "h") / np.timedelta64(1, "s")
hour = np.timedelta64(1, "h") / np.timedelta64(1, "s")
minute = np.timedelta64(1, "m") / np.timedelta64(1, "s")
second = np.timedelta64(1, "s") / np.timedelta64(1, "s")
millisecond = np.timedelta64(1, "ms") / np.timedelta64(1, "s")
microsecond = np.timedelta64(1, "us") / np.timedelta64(1, "s")
nanosecond = np.timedelta64(1, "ns") / np.timedelta64(1, "s")


@pytest.mark.parametrize(
    "duration, sampling_rate, expected",
    [
        (None, None, np.NaN),
        (None, 1000, np.NaN),
        ("", None, np.NaN),
        ("", 1000, np.NaN),
        ("none", None, np.NaN),
        ("none", 1000, np.NaN),
        ("None", None, np.NaN),
        ("None", 1000, np.NaN),
        ("nan", None, np.NaN),
        ("nan", 1000, np.NaN),
        ("NaN", None, np.NaN),
        ("NaN", 1000, np.NaN),
        ("nat", None, np.NaN),
        ("nat", 1000, np.NaN),
        ("NaT", None, np.NaN),
        ("NaT", 1000, np.NaN),
        (np.NaN, None, np.NaN),
        (np.NaN, 1000, np.NaN),
        (pd.NaT, None, np.NaN),
        (pd.NaT, 1000, np.NaN),
        (pd.NA, None, np.NaN),
        (pd.NA, 1000, np.NaN),
        (np.timedelta64("NaT", "s"), None, np.NaN),
        (np.timedelta64("NaT", "s"), 1000, np.NaN),
        ("inf", None, np.inf),
        ("inf", 1000, np.inf),
        ("Inf", None, np.inf),
        ("Inf", 1000, np.inf),
        (np.inf, None, np.inf),
        (np.inf, 1000, np.inf),
        (np.Inf, None, np.inf),
        (np.Inf, 1000, np.inf),
        (2, None, 2.0),
        (2, 1000, 0.002),
        (2.0, None, 2.0),
        (2.0, 1000, 0.002),
        ("s", None, 1.0),
        ("s", 1000, 1.0),
        (" s", None, 1.0),
        (" s", 1000, 1.0),
        ("2s", None, 2.0),
        ("2s", 1000, 2.0),
        ("2s ", None, 2.0),
        ("2s ", 1000, 2.0),
        (" 2s", None, 2.0),
        (" 2s", 1000, 2.0),
        ("2 s", None, 2.0),
        ("2 s", 1000, 2.0),
        ("2  s", None, 2.0),
        ("2  s", 1000, 2.0),
        ("2000ms", None, 2.0),
        ("2000ms", 1000, 2.0),
        ("2000.0ms", None, 2.0),
        ("2000.0ms", 1000, 2.0),
        ("2000 ms", None, 2.0),
        ("2000 ms", 1000, 2.0),
        ("2000.0 ms", None, 2.0),
        ("2000.0 ms", 1000, 2.0),
        ("2000", None, 2000.0),
        ("2000", 1000, 2.0),
        ("2000 ", None, 2000.0),
        ("2000 ", 1000, 2.0),
        ("2000.0", None, 2000.0),
        ("2000.0", 1000, 2.0),
        ("2000.1", None, 2000.1),
        ("2000.1", 1000, 2.0000999999999998),
        ("0.5", None, 0.5),
        ("0.5", 2, 0.25),
        ("3", 1.5, 2.0),
        (np.timedelta64(2, "s"), None, 2.0),
        (np.timedelta64(2, "s"), 1000, 2.0),
        (np.timedelta64(2000, "ms"), None, 2.0),
        (np.timedelta64(2000, "ms"), 1000, 2.0),
        (pd.to_timedelta(2, "s"), None, 2.0),
        (pd.to_timedelta(2, "s"), 1000, 2.0),
        (pd.to_timedelta(2000, "ms"), None, 2.0),
        (pd.to_timedelta(2000, "ms"), 1000, 2.0),
        ("+inf", None, np.inf),
        ("+inf", 1000, np.inf),
        ("+Inf", None, np.inf),
        ("+Inf", 1000, np.inf),
        (+2, None, 2.0),
        (+2, 1000, 0.002),
        (+2.0, None, 2.0),
        (+2.0, 1000, 0.002),
        ("+s", None, 1.0),
        ("+s", 1000, 1.0),
        (" +s", None, 1.0),
        (" +s", 1000, 1.0),
        ("+2s", None, 2.0),
        ("+2s", 1000, 2.0),
        ("+2s ", None, 2.0),
        ("+2s ", 1000, 2.0),
        (" +2s", None, 2.0),
        (" +2s", 1000, 2.0),
        ("+2 s", None, 2.0),
        ("+2 s", 1000, 2.0),
        ("+2  s", None, 2.0),
        ("+2  s", 1000, 2.0),
        ("+2000ms", None, 2.0),
        ("+2000ms", 1000, 2.0),
        ("+2000.0ms", None, 2.0),
        ("+2000.0ms", 1000, 2.0),
        ("+2000 ms", None, 2.0),
        ("+2000 ms", 1000, 2.0),
        ("+2000.0 ms", None, 2.0),
        ("+2000.0 ms", 1000, 2.0),
        ("+2000", None, 2000.0),
        ("+2000", 1000, 2.0),
        ("+2000 ", None, 2000.0),
        ("+2000 ", 1000, 2.0),
        ("+2000.0", None, 2000.0),
        ("+2000.0", 1000, 2.0),
        ("+2000.1", None, 2000.1),
        ("+2000.1", 1000, 2.0000999999999998),
        ("+0.5", None, 0.5),
        ("+0.5", 2, 0.25),
        ("+3", 1.5, 2.0),
        (np.timedelta64(+2, "s"), None, 2.0),
        (np.timedelta64(+2, "s"), 1000, 2.0),
        (np.timedelta64(+2000, "ms"), None, 2.0),
        (np.timedelta64(+2000, "ms"), 1000, 2.0),
        (pd.to_timedelta(+2, "s"), None, 2.0),
        (pd.to_timedelta(+2, "s"), 1000, 2.0),
        (pd.to_timedelta(+2000, "ms"), None, 2.0),
        (pd.to_timedelta(+2000, "ms"), 1000, 2.0),
        ("-inf", None, -np.inf),
        ("-inf", 1000, -np.inf),
        ("-Inf", None, -np.inf),
        ("-Inf", 1000, -np.inf),
        (-2, None, -2.0),
        (-2, 1000, -0.002),
        (-2.0, None, -2.0),
        (-2.0, 1000, -0.002),
        ("-s", None, -1.0),
        ("-s", 1000, -1.0),
        (" -s", None, -1.0),
        (" -s", 1000, -1.0),
        ("-2s", None, -2.0),
        ("-2s", 1000, -2.0),
        ("-2s ", None, -2.0),
        ("-2s ", 1000, -2.0),
        (" -2s", None, -2.0),
        (" -2s", 1000, -2.0),
        ("-2 s", None, -2.0),
        ("-2 s", 1000, -2.0),
        ("-2  s", None, -2.0),
        ("-2  s", 1000, -2.0),
        ("-2000ms", None, -2.0),
        ("-2000ms", 1000, -2.0),
        ("-2000.0ms", None, -2.0),
        ("-2000.0ms", 1000, -2.0),
        ("-2000 ms", None, -2.0),
        ("-2000 ms", 1000, -2.0),
        ("-2000.0 ms", None, -2.0),
        ("-2000.0 ms", 1000, -2.0),
        ("-2000", None, -2000.0),
        ("-2000", 1000, -2.0),
        ("-2000 ", None, -2000.0),
        ("-2000 ", 1000, -2.0),
        ("-2000.0", None, -2000.0),
        ("-2000.0", 1000, -2.0),
        ("-2000.1", None, -2000.1),
        ("-2000.1", 1000, -2.0000999999999998),
        ("-0.5", None, -0.5),
        ("-0.5", 2, -0.25),
        ("-3", 1.5, -2.0),
        (np.timedelta64(-2, "s"), None, -2.0),
        (np.timedelta64(-2, "s"), 1000, -2.0),
        (np.timedelta64(-2000, "ms"), None, -2.0),
        (np.timedelta64(-2000, "ms"), 1000, -2.0),
        (pd.to_timedelta(-2, "s"), None, -2.0),
        (pd.to_timedelta(-2, "s"), 1000, -2.0),
        (pd.to_timedelta(-2000, "ms"), None, -2.0),
        (pd.to_timedelta(-2000, "ms"), 1000, -2.0),
        # week
        ("1W", None, week),
        # day
        ("1D", None, day),
        ("1days", None, day),
        ("1day", None, day),
        # hour
        ("1h", None, hour),
        ("1hours", None, hour),
        ("1hour", None, hour),
        ("1hr", None, hour),
        # minute
        ("1m", None, minute),
        ("1minutes", None, minute),
        ("1minute", None, minute),
        ("1min", None, minute),
        ("1T", None, minute),
        # second
        ("1s", None, second),
        ("1seconds", None, second),
        ("1second", None, second),
        ("1sec", None, second),
        ("1S", None, second),
        # millisecond
        ("1ms", None, millisecond),
        ("1milliseconds", None, millisecond),
        ("1millisecond", None, millisecond),
        ("1millis", None, millisecond),
        ("1milli", None, millisecond),
        ("1L", None, millisecond),
        # microsecond
        ("1us", None, microsecond),
        ("1μs", None, microsecond),
        ("1microseconds", None, microsecond),
        ("1microsecond", None, microsecond),
        ("1micros", None, microsecond),
        ("1micro", None, microsecond),
        ("1U", None, microsecond),
        # nanosecond
        ("1ns", None, nanosecond),
        ("1nanoseconds", None, nanosecond),
        ("1nanosecond", None, nanosecond),
        ("1nanos", None, nanosecond),
        ("1nano", None, nanosecond),
        ("1N", None, nanosecond),
    ],
)
def test_duration_in_seconds(duration, sampling_rate, expected):
    duration_in_seconds = audmath.duration_in_seconds(duration, sampling_rate)
    if np.isnan(expected):
        assert np.isnan(duration_in_seconds)
    else:
        assert duration_in_seconds == expected


@pytest.mark.parametrize(
    "duration, sampling_rate, error, error_msg",
    [
        (
            "2abc",
            None,
            ValueError,
            "The provided unit 'abc' is not known.",
        ),
        (
            "2 abc",
            None,
            ValueError,
            "The provided unit 'abc' is not known.",
        ),
        (
            "2a bc",
            None,
            ValueError,
            (
                "Your given duration '2a bc' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "2.0a bc",
            None,
            ValueError,
            (
                "Your given duration '2.0a bc' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            " ",
            None,
            ValueError,
            (
                "Your given duration ' ' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "  ",
            None,
            ValueError,
            (
                "Your given duration '  ' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "1 0 ms",
            None,
            ValueError,
            (
                "Your given duration '1 0 ms' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "10 m s",
            None,
            ValueError,
            (
                "Your given duration '10 m s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "1 0 m s",
            None,
            ValueError,
            (
                "Your given duration '1 0 m s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
        (
            "2.m5s",
            None,
            ValueError,
            (
                "Your given duration '2.m5s' "
                "is not conform to the <value><unit> pattern."
            ),
        ),
    ],
)
def test_duration_in_seconds_error(duration, sampling_rate, error, error_msg):
    with pytest.raises(error, match=re.escape(error_msg)):
        audmath.duration_in_seconds(duration, sampling_rate)
