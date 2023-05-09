import torch
import torch.nn.functional as F

from kwja.utils.constants import IGNORE_INDEX, MASKED

eps = 1e-6


def _average_loss(
    losses: torch.Tensor,  # (b, *dim)
    mask: torch.Tensor,  # (b, *dim)
) -> torch.Tensor:  # ()
    masked_loss_in_batch = (losses * mask).sum(dim=tuple(range(1, losses.ndim)))  # (b)
    labels_in_batch = mask.sum(dim=tuple(range(1, losses.ndim)))  # (b)
    return (masked_loss_in_batch / (labels_in_batch + eps)).sum() / (labels_in_batch.ne(0).sum() + eps)


def compute_token_mean_loss(
    input_: torch.Tensor,  # (b, *seq, num_classes)
    target: torch.Tensor,  # (b, *seq)
) -> torch.Tensor:  # ()
    batch_size, *_, num_classes = input_.shape
    input_ = input_.view(batch_size, -1, num_classes)  # (b, seq, num_classes)
    target = target.view(batch_size, -1)  # (b, seq)
    losses = F.cross_entropy(input_.transpose(1, 2), target, ignore_index=IGNORE_INDEX, reduction="none")  # (b, seq)
    return _average_loss(losses, target.ne(IGNORE_INDEX))


def compute_multi_label_token_mean_loss(
    input_: torch.Tensor,  # (b, seq, num_features)
    target: torch.Tensor,  # (b, seq, num_features)
) -> torch.Tensor:  # ()
    # binary_cross_entropy は IGNORE_INDEX を渡せない
    losses = F.binary_cross_entropy(input_, target.float(), reduction="none")  # (b, seq, num_features)
    mask: torch.Tensor = target.ne(IGNORE_INDEX)  # (b, seq, num_features)
    # features の軸は和をとる
    losses = (losses * mask).sum(dim=2)  # (b, seq)
    return _average_loss(losses, mask[:, :, 0])


def compute_cohesion_analysis_loss(
    input_: torch.Tensor,  # (b, rel, seq, seq)
    target: torch.Tensor,  # (b, rel, seq, seq)
) -> torch.Tensor:  # ()
    log_softmax = torch.log_softmax(input_, dim=3)  # (b, rel, seq, seq)
    return _average_loss(-log_softmax, target)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logits, torch.full_like(logits, MASKED))
