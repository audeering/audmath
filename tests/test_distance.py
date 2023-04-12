import numpy as np
import pytest
import scipy.spatial

import audmath


@pytest.mark.parametrize(
    'x1, x2, metric',
    [
        (
            [1],
            [0.5],
            'cosine',
        ),
        (
            [1],
            [0.5],
            'euclidean',
        ),
        (
            [[1]],
            [[0.5]],
            'cosine',
        ),
        (
            [[1]],
            [[0.5]],
            'euclidean',
        ),
        (
            [1, 0],
            [0, 1],
            'cosine',
        ),
        (
            [1, 0],
            [0, 1],
            'euclidean',
        ),
        (
            [[1, 0], [0, 1]],
            [1, 1],
            'cosine',
        ),
        (
            [[1, 0], [0, 1]],
            [1, 1],
            'euclidean',
        ),
        (
            [1, 0],
            [[1, 1], [1, 1]],
            'cosine',
        ),
        (
            [1, 0],
            [[1, 1], [1, 1]],
            'euclidean',
        ),
        (
            [[1, 0], [0, 1]],
            [[1, 1], [1, 1]],
            'cosine',
        ),
        (
            [[1, 0], [0, 1]],
            [[1, 1], [1, 1]],
            'euclidean',
        ),
        (
            [[1, 0], [2, 3]],
            [[0, 1], [4, 5]],
            'cosine',
        ),
        (
            [[1, 0], [2, 3]],
            [[0, 1], [4, 5]],
            'euclidean',
        ),
    ]
)
def test_distance(x1, x2, metric):
    # Calculate expected results based on scipy implementations
    x1_expected = np.atleast_2d(np.array(x1))
    x2_expected = np.atleast_2d(np.array(x2))
    if metric == 'euclidean':
        expected = scipy.spatial.distance_matrix(x1_expected, x2_expected)
    elif metric == 'cosine':
        expected = []
        for m in range(x1_expected.shape[0]):
            expected.append(
                [
                    scipy.spatial.distance.cosine(
                        x1_expected[m, :],
                        x2_expected[n, :],
                    )
                    for n in range(x2_expected.shape[0])
                ]
            )
        expected = np.array(expected)
    expected = expected.squeeze()
    if not expected.shape:
        expected = float(expected)

    for u in [x1, np.array(x1)]:
        for v in [x2, np.array(x2)]:
            distance = audmath.distance(u, v, metric=metric)
            np.testing.assert_allclose(distance, expected)


def test_distance_errors():
    error_msg = "metric has to be 'cosine' or 'euclidean', not 'faulty'"
    with pytest.raises(ValueError, match=error_msg):
        audmath.distance([1], [1], metric='faulty')
