import typing


def polyval(
        x: float,
        coefs: typing.Sequence,
) -> float:
    r"""Evaluation of polynomial.

    Args:
        x: input value
        coefs: polynomial coefficients

    Returns:
        evaluated polynomial

    """
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        try:
            ans += coef * x**power
        except OverflowError:  # pragma: nocover
            pass
        power -= 1
    return ans
