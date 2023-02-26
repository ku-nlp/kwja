from pathlib import Path

import torch
from transformers import AutoTokenizer

from kwja.datamodule.datasets.char_dataset import CharDataset
from kwja.evaluators.char_module_metric import CharModuleMetric
from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


def test_char_module_metric() -> None:
    metric = CharModuleMetric()

    path = Path(__file__).absolute().parent.joinpath("char_files")
    tokenizer = AutoTokenizer.from_pretrained("ku-nlp/roberta-base-japanese-char-wwm")
    max_seq_length = 20
    dataset = CharDataset(str(path), tokenizer, max_seq_length)  # denormalize_probability == 0.0
    metric.set_properties(dataset)

    metric.example_ids = torch.arange(len(dataset), dtype=torch.long)

    metric.word_segmentation_predictions = torch.full((len(dataset), max_seq_length), IGNORE_INDEX, dtype=torch.long)
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

    metric.word_norm_op_predictions = torch.full((len(dataset), max_seq_length), IGNORE_INDEX, dtype=torch.long)
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
    metric.word_norm_op_labels = torch.stack([dataset[eid]["word_norm_op_labels"] for eid in metric.example_ids], dim=0)

    digits = 4
    metrics = {k: round(v, digits) for k, v in metric.compute().items()}

    assert metrics["word_segmentation_accuracy"] == round(15 / 16, digits)
    # tp = 5, fp = 1, fn = 2 (span-level)
    assert metrics["word_segmentation_f1"] == round((2 * 5 / 6 * 5 / 7) / (5 / 6 + 5 / 7), digits)

    assert metrics["word_normalization_accuracy"] == round(14 / 16, digits)
    # tp = 3, fp = 2, fn = 0 (other than KEEP)
    assert metrics["word_normalization_f1"] == round((2 * 3 / 5 * 3 / 3) / (3 / 5 + 3 / 3), digits)
    # tp = 11, fp = 2, fn = 0
    assert metrics["word_normalization_f1:K"] == round((2 * 11 / 13 * 11 / 11) / (11 / 13 + 11 / 11), digits)
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:D"] == round((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1), digits)
    # tp = 0, fp = 0, fn = 1
    assert metrics["word_normalization_f1:V"] == 0.0
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:S"] == round((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1), digits)
    # tp = 0, fp = 0, fn = 1
    assert metrics["word_normalization_f1:P"] == 0.0
    # tp = 1, fp = 0, fn = 0
    assert metrics["word_normalization_f1:E"] == round((2 * 1 / 1 * 1 / 1) / (1 / 1 + 1 / 1), digits)
