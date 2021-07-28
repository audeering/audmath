import typing

import numpy as np

from audmath.core.utils import polyval


def inverse_normal_distribution(
    y: typing.Union[float, typing.Sequence[float], np.ndarray],
) -> typing.Union[float, np.ndarray]:
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
    Douglas Thor `implemented it in pure Python`_ under GPL-3 by.

    The output is identical to the implementation
    provided by :func:`scipy.special.ndtri`,
    and :func:`scipy.stats.norm.ppf`.

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
    func = np.vectorize(_ndtri, otypes=[np.float])
    x = func(y)
    if (
            x.ndim == 0
            or x.ndim == 1 and len(x) < 2
    ):
        x = float(x)
    return x


def _ndtri(
        y: float,
) -> float:
    r"""None vectorized version of ndtri."""
    # Return for non-suited values
    if y == 0:
        return -np.Inf
    if y == 1:
        return np.Inf
    if y < 0 or y > 1:
        return np.NaN

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

    switch_sign = True

    if y > (1 - EXP_NEG2):  # y > 0.864...
        y = 1.0 - y
        switch_sign = False

    # Case where we don't need high precision
    if y > EXP_NEG2:  # y > 0.135...
        y -= 0.5
        y2 = y ** 2
        x = y + y * (y2 * polyval(y2, P0) / polyval(y2, Q0))
        x = x * ROOT_2PI
        return x

    x = np.sqrt(-2.0 * np.log(y))
    x0 = x - np.log(x) / x
    z = 1.0 / x
    if x < 8.0:  # y > exp(-32) = 1.2664165549e-14
        x1 = z * polyval(z, P1) / polyval(z, Q1)
    else:
        x1 = z * polyval(z, P2) / polyval(z, Q2)

    x = x0 - x1
    if switch_sign:
        x = -x

    return x
