from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import CharDataset
from kwja.metrics import CharModuleMetric
from kwja.utils.constants import IGNORE_INDEX, SENT_SEGMENTATION_TAGS, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


def test_char_module_metric(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase) -> None:
    path = data_dir / "datasets" / "char_files"
    max_seq_length = 20
    denormalize_probability = 0.0
    dataset = CharDataset(str(path), char_tokenizer, max_seq_length, denormalize_probability)

    metric = CharModuleMetric(max_seq_length)
    metric.set_properties({"dataset": dataset})
    metric.update(
        {
            "example_ids": torch.empty(0),  # dummy
            "sent_segmentation_predictions": torch.empty(0),
            "word_segmentation_predictions": torch.empty(0),
            "word_norm_op_predictions": torch.empty(0),
            "word_norm_op_labels": torch.empty(0),
        }
    )

    num_examples = len(dataset)
    metric.example_ids = torch.arange(num_examples, dtype=torch.long)

    metric.sent_segmentation_predictions = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
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
    metric.sent_segmentation_predictions[1, 9] = SENT_SEGMENTATION_TAGS.index("B")  # 〜 (gold = "I")

    metric.word_segmentation_predictions = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    # [:, 0] = [CLS]
    metric.word_segmentation_predictions[0, 1] = WORD_SEGMENTATION_TAGS.index("B")  # 花
    metric.word_segmentation_predictions[0, 2] = WORD_SEGMENTATION_TAGS.index("I")  # 咲
    metric.word_segmentation_predictions[0, 3] = WORD_SEGMENTATION_TAGS.index("B")  # ガ
    metric.word_segmentation_predictions[0, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ニ
    metric.word_segmentation_predictions[0, 5] = WORD_SEGMENTATION_TAGS.index("B")  # を
    metric.word_segmentation_predictions[0, 6] = WORD_SEGMENTATION_TAGS.index("B")  # 買
    metric.word_segmentation_predictions[0, 7] = WORD_SEGMENTATION_TAGS.index("I")  # ぅ
    metric.word_segmentation_predictions[1, 1] = WORD_SEGMENTATION_TAGS.index("B")  # う
    metric.word_segmentation_predictions[1, 2] = WORD_SEGMENTATION_TAGS.index("I")  # ま
    metric.word_segmentation_predictions[1, 3] = WORD_SEGMENTATION_TAGS.index("I")  # そ (gold = "B")
    metric.word_segmentation_predictions[1, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ー
    metric.word_segmentation_predictions[1, 5] = WORD_SEGMENTATION_TAGS.index("I")  # で
    metric.word_segmentation_predictions[1, 6] = WORD_SEGMENTATION_TAGS.index("I")  # す
    metric.word_segmentation_predictions[1, 7] = WORD_SEGMENTATION_TAGS.index("B")  # ね
    metric.word_segmentation_predictions[1, 8] = WORD_SEGMENTATION_TAGS.index("I")  # 〜
    metric.word_segmentation_predictions[1, 9] = WORD_SEGMENTATION_TAGS.index("I")  # 〜

    metric.word_norm_op_predictions = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    # [:, 0] = [CLS]
    metric.word_norm_op_predictions[0, 1] = WORD_NORM_OP_TAGS.index("K")  # 花
    metric.word_norm_op_predictions[0, 2] = WORD_NORM_OP_TAGS.index("K")  # 咲
    metric.word_norm_op_predictions[0, 3] = WORD_NORM_OP_TAGS.index("K")  # ガ (gold = "V")
    metric.word_norm_op_predictions[0, 4] = WORD_NORM_OP_TAGS.index("K")  # ニ
    metric.word_norm_op_predictions[0, 5] = WORD_NORM_OP_TAGS.index("K")  # を
    metric.word_norm_op_predictions[0, 6] = WORD_NORM_OP_TAGS.index("K")  # 買
    metric.word_norm_op_predictions[0, 7] = WORD_NORM_OP_TAGS.index("S")  # ぅ
    metric.word_norm_op_predictions[1, 1] = WORD_NORM_OP_TAGS.index("K")  # う
    metric.word_norm_op_predictions[1, 2] = WORD_NORM_OP_TAGS.index("K")  # ま
    metric.word_norm_op_predictions[1, 3] = WORD_NORM_OP_TAGS.index("K")  # そ
    metric.word_norm_op_predictions[1, 4] = WORD_NORM_OP_TAGS.index("K")  # ー (gold = "P")
    metric.word_norm_op_predictions[1, 5] = WORD_NORM_OP_TAGS.index("K")  # で
    metric.word_norm_op_predictions[1, 6] = WORD_NORM_OP_TAGS.index("K")  # す
    metric.word_norm_op_predictions[1, 7] = WORD_NORM_OP_TAGS.index("K")  # ね
    metric.word_norm_op_predictions[1, 8] = WORD_NORM_OP_TAGS.index("E")  # 〜
    metric.word_norm_op_predictions[1, 9] = WORD_NORM_OP_TAGS.index("D")  # 〜
    metric.word_norm_op_labels = torch.stack(
        [torch.as_tensor(dataset[eid].word_norm_op_labels) for eid in metric.example_ids], dim=0
    )

    metrics = metric.compute()

    assert metrics["sent_segmentation_accuracy"] == pytest.approx(15 / 16)
    # tp = 1, fp = 2, fn = 1 (span-level)
    assert metrics["sent_segmentation_f1"] == pytest.approx((2 * 1 / 3 * 1 / 2) / (1 / 3 + 1 / 2))

    assert metrics["word_segmentation_accuracy"] == pytest.approx(15 / 16)
    # tp = 5, fp = 1, fn = 2 (span-level)
    assert metrics["word_segmentation_f1"] == pytest.approx((2 * 5 / 6 * 5 / 7) / (5 / 6 + 5 / 7))

    assert metrics["word_normalization_accuracy"] == pytest.approx(14 / 16)
    # tp = 3, fp = 2, fn = 0 (other than KEEP)
    assert metrics["word_normalization_f1"] == pytest.approx((2 * 3 / 5 * 3 / 3) / (3 / 5 + 3 / 3))
    # tp = 11, fp = 2, fn = 0
    assert metrics["word_normalization_f1:K"] == pytest.approx((2 * 11 / 13 * 11 / 11) / (11 / 13 + 11 / 11))
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:D"] == pytest.approx((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1))
    # tp = 0, fp = 0, fn = 1
    assert metrics["word_normalization_f1:V"] == pytest.approx(0.0)
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:S"] == pytest.approx((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1))
    # tp = 0, fp = 0, fn = 1
    assert metrics["word_normalization_f1:P"] == pytest.approx(0.0)
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:E"] == pytest.approx((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1))
