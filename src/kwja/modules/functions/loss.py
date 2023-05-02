import torch
import torch.nn.functional as F

from kwja.utils.constants import IGNORE_INDEX, MASKED

eps = 1e-6


def compute_token_mean_loss(
    input_: torch.Tensor,  # (b, *seq, num_classes)
    target: torch.Tensor,  # (b, *seq)
) -> torch.Tensor:  # ()
    batch_size, *seq, num_classes = input_.shape
    input_ = input_.view(batch_size, -1, num_classes)  # (b, seq, num_classes)
    target = target.view(batch_size, -1)  # (b, seq)
    losses = F.cross_entropy(input_.transpose(1, 2), target, ignore_index=IGNORE_INDEX, reduction="none")  # (b, seq)
    mask: torch.Tensor = target.ne(IGNORE_INDEX)  # (b, seq)
    return ((losses * mask).sum(dim=1) / (mask.sum(dim=1) + eps)).mean()


def compute_multi_label_token_mean_loss(
    input_: torch.Tensor,  # (b, seq, num_features)
    target: torch.Tensor,  # (b, seq, num_features)
) -> torch.Tensor:  # ()
    mask: torch.Tensor = target.ne(IGNORE_INDEX)  # (b, seq)
    # binary_cross_entropy は IGNORE_INDEX を渡せない
    losses = F.binary_cross_entropy(input_ * mask, target.float() * mask, reduction="none")  # (b, seq, num_features)
    # batch と sequence の軸は平均を、 label の軸は和をとる
    losses = (losses * mask).sum(dim=2)  # (b, seq)
    return (losses.sum(dim=1) / (mask[:, :, 0].sum(dim=1) + eps)).mean()


def compute_cohesion_analysis_loss(
    input_: torch.Tensor,  # (b, rel, seq, seq)
    target: torch.Tensor,  # (b, rel, seq, seq)
) -> torch.Tensor:  # ()
    log_softmax = torch.log_softmax(input_, dim=3)  # (b, rel, seq, seq)
    return ((-log_softmax * target).sum((1, 2, 3)) / (target.sum((1, 2, 3)) + eps)).mean()


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logits, torch.full_like(logits, MASKED))
