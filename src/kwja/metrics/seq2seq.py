import torch

from kwja.metrics.base import BaseModuleMetric


class Seq2SeqModuleMetric(BaseModuleMetric):
    STATE_NAMES = ("loss",)

    def __init__(self, max_seq_length: int) -> None:
        super().__init__(max_seq_length)
        self.loss: torch.Tensor

    def _pad(self, kwargs: dict[str, torch.Tensor]) -> None:
        pass

    def compute(self) -> dict[str, float]:
        # The loss is a scalar per batch; distributed sync concatenates the list into a tensor beforehand.
        loss = self.loss if isinstance(self.loss, torch.Tensor) else torch.stack(self.loss)
        return {"seq2seq_loss": loss.mean().item()}
