import collections
import typing
import warnings

import numpy as np

from audmath.core.utils import polyval


FADEIN_SHAPES = [
    'tukey',
    'kaiser',
    'linear',
    'exponential',
    'logarithmic',
]


def db(
        x: typing.Union[int, float, typing.Sequence, np.ndarray],
        *,
        bottom: typing.Union[int, float] = -120,
) -> typing.Union[np.floating, np.ndarray]:
    r"""Convert value to decibels.

    The decibel of a value :math:`x \in \R`
    is given by

    .. math::

        \text{db}(x) = \begin{cases}
            20 \log_{10} x,
                & \text{if } x > 10^\frac{\text{bottom}}{20} \\
            \text{bottom},
                & \text{else}
        \end{cases}

    where :math:`\text{bottom}` is provided
    by the argument of same name.

    Args:
        x: input value(s)
        bottom: minimum decibel value
            returned for very low input values.
            If set to ``None``
            it will return ``-np.Inf``
            for values equal or less than 0

    Returns:
        input value(s) in dB

    Example:
        >>> db(1)
        0.0
        >>> db(0)
        -120.0
        >>> db(2)
        6.020599913279624
        >>> db([0, 1])
        array([-120.,    0.])

    """
    if bottom is None:
        min_value = 0
        bottom = -np.Inf
    else:
        bottom = np.float64(bottom)
        min_value = 10 ** (bottom / 20)

    if not isinstance(x, (collections.abc.Sequence, np.ndarray)):
        if x <= min_value:
            return bottom
        else:
            return 20 * np.log10(x)

    x = np.array(x)
    if x.size == 0:
        return x

    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)

    mask = (x <= min_value)
    x[mask] = bottom
    x[~mask] = 20 * np.log10(x[~mask])

    return x


def fadein(
    samples: int,
    shape: str = 'tukey',
    level: float = -120.,
    bottom: typing.Union[int, float] = -120,
) -> np.ndarray:
    r"""Fade-in half-window.

    A fade-in is a gradual increase in amplitude
    of a signal.
    If ``level`` <= ``bottom``
    the fadein will start from 0,
    otherwise from the provided level.

    The shape of the fade-in and fade-out
    is selected via ``in_shape`` and ``out_shape``.
    The following figure shows all available shapes
    by the example of a fade-in.

    .. plot::

        import audmath
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import numpy as np
        import seaborn as sns

        for shape in audmath.core.api.FADEIN_SHAPES:
            win = audmath.fadein(100, shape=shape)
            win = np.concatenate([win, np.array([1.])])
            plt.plot(win, label=shape)
        plt.ylabel('Magnitude')
        plt.xlabel('Fade-in Length')
        plt.grid(alpha=0.4)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis=u'both', which=u'both',length=0)
        plt.xlim([-1.2, 100.3])
        plt.ylim([-0.02, 1])
        sns.despine(left=True, bottom=True)
        # Put a legend to the top right of the current axis
        plt.legend()
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # Adjsut image size to contain outside legend
        fig = plt.gcf()
        fig.set_size_inches(6.4, 3.84)
        plt.tight_layout()

    If at least 2 samples are requested,
    the fade-in half-window will always start at 0
    or the value provided by ``level`` and ``bottom``
    and end at 1.

    Args:
        samples: length of fade-in half-window
        shape: shape of fade-in half-window
        level: start level in decibel of fade-in
        bottom: minimum level in decibel
            above which the half-window
            will not start at a value of 0

    Returns:
        fade-in half-window

    Raises:
        ValueError: if requested ``shape`` is not supported

    Example:
        >>> fadein(5)
        array([0.        , 0.14644661, 0.5       , 0.85355339, 1.        ])

    """
    if shape not in FADEIN_SHAPES:
        raise ValueError(
            "shape has to be one of the following: "
            f"{(', ').join(FADEIN_SHAPES)},"
            f"not '{shape}'."
        )
    if samples < 2:
        win = np.arange(samples)
    elif shape == 'linear':
        win = np.arange(samples) / (samples - 1)
    elif shape == 'kaiser':
        # Kaiser windows as approximation of DPSS window
        # as often used for tapering windows
        win = np.kaiser(2 * (samples - 1), beta=14)[:(samples - 1)]
        # Ensure first entry is 0
        win[0] = 0
        # Add 1 at the end
        win = np.concatenate([win, np.array([1])])
    elif shape == 'tukey':
        # Tukey window,
        # which is also often used as tapering window
        # 1/2 * (1 - cos(2pi n / (4N alpha)))
        x = np.arange(samples)
        alpha = 0.5
        width = 4 * (samples - 1) * alpha
        win = 0.5 * (1 - np.cos(2 * np.pi * x / width))
    elif shape == 'exponential':
        x = np.arange(samples)
        win = (np.exp(x) - 1) / (np.exp(samples - 1) - 1)
    elif shape == 'logarithmic':
        x = np.arange(samples)
        win = np.log10(x + 1) / np.log10(samples)
    offset = inverse_db(level, bottom=bottom)
    win = win * (1 - offset) + offset
    return win


def inverse_db(
        y: typing.Union[int, float, typing.Sequence, np.ndarray],
        *,
        bottom: typing.Union[int, float] = -120,
) -> typing.Union[np.floating, np.ndarray]:
    r"""Convert decibels to amplitude value.

    The inverse of a value :math:`y \in \R`
    provided in decibel
    is given by

    .. math::

        \text{inverse\_db}(y) = \begin{cases}
            10^\frac{y}{20},
                & \text{if } y > \text{bottom} \\
            0,
                & \text{else}
        \end{cases}

    where :math:`\text{bottom}` is provided
    by the argument of same name.

    Args:
        y: input signal in decibels
        bottom: minimum decibel value
            which should be converted.
            Lower values will be set to 0.
            If set to ``None``
            it will return 0
            only for input values of ``-np.Inf``

    Returns:
        input signal

    Example:
        >>> inverse_db(0)
        1.0
        >>> inverse_db(-120)
        0.0
        >>> inverse_db(-3)
        0.7079457843841379
        >>> inverse_db([-120, 0])
        array([0., 1.])

    """
    min_value = 0.
    if bottom is None:
        bottom = -np.Inf

    if not isinstance(y, (collections.abc.Sequence, np.ndarray)):
        if y <= bottom:
            return min_value
        else:
            return np.power(10., y / 20.)

    y = np.array(y)
    if y.size == 0:
        return y

    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float64)

    mask = (y <= bottom)
    y[mask] = min_value
    y[~mask] = np.power(10., y[~mask] / 20.)
    return y


def inverse_normal_distribution(
    y: typing.Union[int, float, typing.Sequence, np.ndarray],
) -> typing.Union[np.floating, np.ndarray]:
    r"""Inverse normal distribution.

    Returns the argument :math:`x`
    for which the area under the Gaussian probability density function
    is equal to :math:`y`.
    It returns :math:`\text{nan}`
    if :math:`y \notin [0, 1]`.

    The area under the Gaussian probability density function is given by:

    .. math::

        \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x \exp(-t^2 / 2)\,\text{d}t

    This function is a :mod:`numpy` port
    of the `Cephes C code`_.
    Douglas Thor `implemented it in pure Python`_ under GPL-3.

    The output is identical to the implementation
    provided by :func:`scipy.special.ndtri`,
    and :func:`scipy.stats.norm.ppf`,
    and allows you
    to avoid installing
    and importing :mod:`scipy`.
    :func:`audmath.inverse_normal_distribution`
    is slower for large arrays
    as the following comparison of execution times
    on a standard PC show.

    .. table::

        ========== ======= =======
        Samples    scipy   audmath
        ========== ======= =======
            10.000   0.00s   0.01s
           100.000   0.00s   0.09s
         1.000.000   0.02s   0.88s
        10.000.000   0.25s   9.30s
        ========== ======= =======


    .. _Cephes C code: https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/cprob/ndtri.c
    .. _implemented it in pure Python: https://github.com/dougthor42/PyErf/blob/cf38a2c62556cbd4927c9b3f5523f39b6a492472/pyerf/pyerf.py#L183-L287

    Args:
        y: input value

    Returns:
        inverted input

    Example:
        >>> inverse_normal_distribution([0.05, 0.4, 0.6, 0.95])
        array([-1.64485363, -0.2533471 , 0.2533471 , 1.64485363])

    """  # noqa: E501
    if isinstance(y, np.ndarray):
        y = y.copy()
    y = np.atleast_1d(y)
    x = np.zeros(y.shape)
    switch_sign = np.ones(y.shape)

    # Handle edge cases
    idx1 = y == 0
    x[idx1] = -np.Inf
    idx2 = y == 1
    x[idx2] = np.Inf
    idx3 = y < 0
    x[idx3] = np.NaN
    idx4 = y > 1
    x[idx4] = np.NaN
    non_valid = np.array([any(i) for i in zip(idx1, idx2, idx3, idx4)])

    # Return if no other values are left
    if non_valid.sum() == len(x):
        return np.float64(x)

    switch_sign[non_valid] = 0

    # Constants to avoid recalculation
    ROOT_2PI = np.sqrt(2 * np.pi)
    EXP_NEG2 = np.exp(-2)

    # Approximation for 0 <= |y - 0.5| <= 3/8
    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]
    Q0 = [
        1.0,
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8,
    # i.e. y between exp(-2) = .135 and exp(-32) = 1.27e-14
    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]

    Q1 = [
        1.0,
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64,
    # i.e. y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890
    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]

    Q2 = [
        1.0,
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]

    idx1 = y > (1 - EXP_NEG2)  # y > 0.864...
    idx = np.where(non_valid, False, idx1)
    y[idx] = 1.0 - y[idx]
    switch_sign[idx] = 0

    # Case where we don't need high precision
    idx2 = y > EXP_NEG2  # y > 0.135...
    idx = np.where(non_valid, False, idx2)
    y[idx] = y[idx] - 0.5
    y2 = y[idx] ** 2
    x[idx] = y[idx] + y[idx] * (y2 * polyval(y2, P0) / polyval(y2, Q0))
    x[idx] = x[idx] * ROOT_2PI
    switch_sign[idx] = 0

    idx3 = ~idx2
    idx = np.where(non_valid, False, idx3)
    x[idx] = np.sqrt(-2.0 * np.log(y[idx]))
    x0 = x[idx] - np.log(x[idx]) / x[idx]
    z = 1.0 / x[idx]
    x1 = np.where(
        x[idx] < 8.0,  # y > exp(-32) = 1.2664165549e-14
        z * polyval(z, P1) / polyval(z, Q1),
        z * polyval(z, P2) / polyval(z, Q2),
    )
    x[idx] = x0 - x1

    x = np.where(switch_sign == 1, -1 * x, x)

    return np.float64(x)


def rms(
        x: typing.Union[int, float, typing.Sequence, np.ndarray],
        *,
        axis: typing.Union[int, typing.Tuple[int]] = None,
        keepdims: bool = False,
) -> typing.Union[np.floating, np.ndarray]:
    r"""Root mean square.

    The root mean square
    for a signal of length :math:`N`
    is given by

    .. math::

        \sqrt{\frac{1}{N} \sum_{n=1}^N x_n^2}

    where :math:`x_n` is the value
    of a single sample
    of the signal.

    For an empty signal
    0 is returned.

    Args:
        x: input signal
        axis: axis or axes
            along which the root mean squares are computed.
            The default is to compute the root mean square
            of the flattened signal
        keepdims: if this is set to ``True``,
            the axes which are reduced
            are left in the result
            as dimensions with size one

    Returns:
        root mean square of input signal

    Example:
        >>> rms([])
        0.0
        >>> rms([0, 1])
        0.7071067811865476
        >>> rms([[0, 1], [0, 1]])
        0.7071067811865476
        >>> rms([[0, 1], [0, 1]], keepdims=True)
        array([[0.70710678]])
        >>> rms([[0, 1], [0, 1]], axis=1)
        array([0.70710678, 0.70710678])

    """
    x = np.array(x)
    if x.size == 0:
        return np.float64(0.0)
    return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims))
