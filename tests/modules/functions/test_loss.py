import pytest
import torch

from kwja.modules.functions.loss import compute_multi_label_token_mean_loss, compute_token_mean_loss, mask_logits
from kwja.utils.constants import MASKED


@pytest.mark.parametrize(
    "input_, target, expected",
    [
        (
            torch.tensor([[1.0, 0.0]], dtype=torch.float),  # (1, 2)
            torch.tensor([[0]], dtype=torch.long),  # (1,)
            torch.tensor(0.3132616875182228, dtype=torch.float),  # (,)
        ),
    ],
)
def test_compute_token_mean_loss(input_: torch.Tensor, target: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isclose(compute_token_mean_loss(input_, target), expected)


@pytest.mark.parametrize(
    "input_, target, expected",
    [
        (
            torch.tensor([[[0.5, 0.5], [1.0, 0.5]]], dtype=torch.float),  # (1, 2, 2)
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float),  # (1, 2, 2)
            torch.tensor(1.0397207708399179, dtype=torch.float),
        ),
    ],
)
def test_compute_multi_label_token_mean_loss(
    input_: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert torch.isclose(compute_multi_label_token_mean_loss(input_, target), expected)


@pytest.mark.parametrize(
    "logits, mask, expected",
    [
        (
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),  # (3,)
            torch.tensor([True, False, False], dtype=torch.bool),  # (3,)
            torch.tensor([1.0, MASKED, MASKED], dtype=torch.float),  # (3,)
        )
    ],
)
def test_mask_logits(logits: torch.Tensor, mask: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.allclose(mask_logits(logits, mask), expected)
