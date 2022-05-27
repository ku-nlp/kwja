import torch

from jula.models.models.pooling import PoolingStrategy, pool_subwords

eps = 1e-3

# (b, word, seq) = (2, 3, 5)
subword_map = torch.tensor(
    [
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
    ],
    dtype=torch.bool,
)


# (b, seq, hid) = (2, 6, 4)
sequence_output = torch.tensor(
    [
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [-41, -42, -43, -44],
            [-51, -52, -53, -54],
            [-61, -62, -63, -64],
        ],
        [
            [7, 0, 7, 5],
            [0, 8, 4, 0],
            [5, 0, 8, 2],
            [3, 2, 1, 5],
            [9, 8, 7, 9],
            [1, 0, 2, 3],
        ],
    ],
    dtype=torch.float,
)


def test_first_pooling():
    # (b, word, hid) = (2, 3, 4)
    pooled_output = pool_subwords(sequence_output, subword_map, PoolingStrategy.FIRST)
    expected_output = torch.tensor(
        [
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
            ],
            [
                [0, 8, 4, 0],
                [3, 2, 1, 5],
                [1, 0, 2, 3],
            ],
        ],
        dtype=torch.float,
    )
    assert (pooled_output - expected_output).abs().sum().item() < eps


def test_last_pooling():
    # (b, word, hid) = (2, 3, 4)
    pooled_output = pool_subwords(sequence_output, subword_map, PoolingStrategy.LAST)
    expected_output = torch.tensor(
        [
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [-61, -62, -63, -64],
            ],
            [
                [5, 0, 8, 2],
                [9, 8, 7, 9],
                [1, 0, 2, 3],
            ],
        ],
        dtype=torch.float,
    )
    assert (pooled_output - expected_output).abs().sum().item() < eps


def test_max_pooling():
    # (b, word, hid) = (2, 3, 4)
    pooled_output = pool_subwords(sequence_output, subword_map, PoolingStrategy.MAX)
    expected_output = torch.tensor(
        [
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
            ],
            [
                [5, 8, 8, 2],
                [9, 8, 7, 9],
                [1, 0, 2, 3],
            ],
        ],
        dtype=torch.float,
    )
    assert (pooled_output - expected_output).abs().sum().item() < eps


def test_average_pooling():
    # (b, word, hid) = (2, 3, 4)
    pooled_output = pool_subwords(sequence_output, subword_map, PoolingStrategy.AVE)
    expected_output = torch.tensor(
        [
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [-30.5, -31, -31.5, -32],
            ],
            [
                [2.5, 4, 6, 1],
                [6, 5, 4, 7],
                [1, 0, 2, 3],
            ],
        ],
        dtype=torch.float,
    )
    assert (pooled_output - expected_output).abs().sum().item() < eps
