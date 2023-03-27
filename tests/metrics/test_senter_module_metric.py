from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import SenterDataset
from kwja.metrics import SenterModuleMetric
from kwja.utils.constants import IGNORE_INDEX, SENT_SEGMENTATION_TAGS


def test_char_module_metric(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase) -> None:
    metric = SenterModuleMetric()

    path = fixture_data_dir / "datasets" / "char_files"
    max_seq_length = 20
    dataset = SenterDataset(str(path), char_tokenizer, max_seq_length=max_seq_length)
    metric.set_properties({"dataset": dataset})

    metric.update(
        {
            "example_ids": torch.empty(0),  # dummy
            "sent_segmentation_predictions": torch.empty(0),
            "word_norm_op_predictions": torch.empty(0),
            "word_norm_op_labels": torch.empty(0),
        }
    )

    num_examples = len(dataset)
    metric.example_ids = torch.arange(num_examples, dtype=torch.long)

    metric.sent_segmentation_predictions = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    # [:, 0] = [CLS]
    metric.sent_segmentation_predictions[0, 1] = SENT_SEGMENTATION_TAGS.index("B")  # 花
    metric.sent_segmentation_predictions[0, 2] = SENT_SEGMENTATION_TAGS.index("I")  # 咲
    metric.sent_segmentation_predictions[0, 3] = SENT_SEGMENTATION_TAGS.index("I")  # ガ
    metric.sent_segmentation_predictions[0, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ニ
    metric.sent_segmentation_predictions[0, 5] = SENT_SEGMENTATION_TAGS.index("I")  # を
    metric.sent_segmentation_predictions[0, 6] = SENT_SEGMENTATION_TAGS.index("I")  # 買
    metric.sent_segmentation_predictions[0, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ぅ
    metric.sent_segmentation_predictions[1, 1] = SENT_SEGMENTATION_TAGS.index("B")  # う
    metric.sent_segmentation_predictions[1, 2] = SENT_SEGMENTATION_TAGS.index("I")  # ま
    metric.sent_segmentation_predictions[1, 3] = SENT_SEGMENTATION_TAGS.index("I")  # そ
    metric.sent_segmentation_predictions[1, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ー
    metric.sent_segmentation_predictions[1, 5] = SENT_SEGMENTATION_TAGS.index("I")  # で
    metric.sent_segmentation_predictions[1, 6] = SENT_SEGMENTATION_TAGS.index("I")  # す
    metric.sent_segmentation_predictions[1, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ね
    metric.sent_segmentation_predictions[1, 8] = SENT_SEGMENTATION_TAGS.index("I")  # 〜
    metric.sent_segmentation_predictions[1, 9] = SENT_SEGMENTATION_TAGS.index("B")  # 〜 (gold: I)

    metrics = metric.compute()

    assert metrics["sent_segmentation_accuracy"] == pytest.approx(15 / 16)
    # tp = 1, fp = 2, fn = 2 (span-level)
    assert metrics["sent_segmentation_f1"] == pytest.approx((2 * 1 / 3 * 1 / 2) / (1 / 3 + 1 / 2))
