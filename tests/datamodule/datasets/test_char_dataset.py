from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import CharDataset
from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


def test_init(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "char_files"
    _ = CharDataset(str(path), char_tokenizer, max_seq_length=512, denormalize_probability=0.0)


def test_getitem(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "char_files"
    max_seq_length: int = 512
    dataset = CharDataset(str(path), char_tokenizer, max_seq_length, denormalize_probability=0.0)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.word_segmentation_labels) == max_seq_length
        assert len(feature.word_norm_op_labels) == max_seq_length


def test_encode(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "char_files"
    max_seq_length = 512
    dataset = CharDataset(str(path), char_tokenizer, max_seq_length, denormalize_probability=0.0)
    num_examples = len(dataset)

    word_segmentation_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    word_segmentation_labels[0, 1] = WORD_SEGMENTATION_TAGS.index("B")  # 花
    word_segmentation_labels[0, 2] = WORD_SEGMENTATION_TAGS.index("I")  # 咲
    word_segmentation_labels[0, 3] = WORD_SEGMENTATION_TAGS.index("B")  # ガ
    word_segmentation_labels[0, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ニ
    word_segmentation_labels[0, 5] = WORD_SEGMENTATION_TAGS.index("B")  # を
    word_segmentation_labels[0, 6] = WORD_SEGMENTATION_TAGS.index("B")  # 買
    word_segmentation_labels[0, 7] = WORD_SEGMENTATION_TAGS.index("I")  # ぅ
    word_segmentation_labels[1, 1] = WORD_SEGMENTATION_TAGS.index("B")  # う
    word_segmentation_labels[1, 2] = WORD_SEGMENTATION_TAGS.index("I")  # ま
    word_segmentation_labels[1, 3] = WORD_SEGMENTATION_TAGS.index("B")  # そ
    word_segmentation_labels[1, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ー
    word_segmentation_labels[1, 5] = WORD_SEGMENTATION_TAGS.index("I")  # で
    word_segmentation_labels[1, 6] = WORD_SEGMENTATION_TAGS.index("I")  # す
    word_segmentation_labels[1, 7] = WORD_SEGMENTATION_TAGS.index("B")  # ね
    word_segmentation_labels[1, 8] = WORD_SEGMENTATION_TAGS.index("I")  # 〜
    word_segmentation_labels[1, 9] = WORD_SEGMENTATION_TAGS.index("I")  # 〜
    assert dataset[0].word_segmentation_labels == word_segmentation_labels[0].tolist()
    assert dataset[1].word_segmentation_labels == word_segmentation_labels[1].tolist()

    word_norm_op_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    word_norm_op_labels[0, 1] = WORD_NORM_OP_TAGS.index("K")  # 花
    word_norm_op_labels[0, 2] = WORD_NORM_OP_TAGS.index("K")  # 咲
    word_norm_op_labels[0, 3] = WORD_NORM_OP_TAGS.index("V")  # ガ
    word_norm_op_labels[0, 4] = WORD_NORM_OP_TAGS.index("K")  # ニ
    word_norm_op_labels[0, 5] = WORD_NORM_OP_TAGS.index("K")  # を
    word_norm_op_labels[0, 6] = WORD_NORM_OP_TAGS.index("K")  # 買
    word_norm_op_labels[0, 7] = WORD_NORM_OP_TAGS.index("S")  # ぅ
    word_norm_op_labels[1, 1] = WORD_NORM_OP_TAGS.index("K")  # う
    word_norm_op_labels[1, 2] = WORD_NORM_OP_TAGS.index("K")  # ま
    word_norm_op_labels[1, 3] = WORD_NORM_OP_TAGS.index("K")  # そ
    word_norm_op_labels[1, 4] = WORD_NORM_OP_TAGS.index("P")  # ー
    word_norm_op_labels[1, 5] = WORD_NORM_OP_TAGS.index("K")  # で
    word_norm_op_labels[1, 6] = WORD_NORM_OP_TAGS.index("K")  # す
    word_norm_op_labels[1, 7] = WORD_NORM_OP_TAGS.index("K")  # ね
    word_norm_op_labels[1, 8] = WORD_NORM_OP_TAGS.index("E")  # 〜
    word_norm_op_labels[1, 9] = WORD_NORM_OP_TAGS.index("D")  # 〜
    assert dataset[0].word_norm_op_labels == word_norm_op_labels[0].tolist()
    assert dataset[1].word_norm_op_labels == word_norm_op_labels[1].tolist()
