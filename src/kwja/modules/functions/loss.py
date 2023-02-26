import torch
import torch.nn.functional as F

from kwja.utils.constants import IGNORE_INDEX, MASKED


def compute_token_mean_loss(input_: torch.Tensor, target: torch.Tensor):
    return F.cross_entropy(
        input=input_.view(-1, input_.size(-1)),
        target=target.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def compute_multi_label_token_mean_loss(input_: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    num_units = mask[:, :, 0].sum(dim=1)  # (b, )
    # binary_cross_entropy は IGNORE_INDEX を渡せない
    losses = F.binary_cross_entropy(input=input_, target=target, reduction="none")
    # batch と sequence の軸は平均を、 label の軸は和をとる
    losses = losses.sum(dim=1).sum(dim=1)  # (b, seq, num_features) -> (b, num_features) -> (b, )
    loss = (losses / num_units).mean()
    return loss


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logits, torch.full_like(logits, MASKED))
