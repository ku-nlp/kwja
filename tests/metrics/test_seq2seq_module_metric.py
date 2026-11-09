import pytest
import torch

from kwja.metrics import Seq2SeqModuleMetric


def test_seq2seq_module_metric_averages_batch_losses() -> None:
    metric = Seq2SeqModuleMetric(max_seq_length=8)
    metric.update({"loss": torch.tensor(1.0)})
    metric.update({"loss": torch.tensor(3.0)})
    assert metric.compute() == {"seq2seq_loss": pytest.approx(2.0)}
