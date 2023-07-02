from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import SenterDataset
from kwja.utils.constants import IGNORE_INDEX, SENT_SEGMENTATION_TAGS


def test_init(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "char_files"
    _ = SenterDataset(str(path), char_tokenizer, max_seq_length=512)


def test_getitem(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "char_files"
    max_seq_length: int = 512
    dataset = SenterDataset(str(path), char_tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.sent_segmentation_labels) == max_seq_length


def test_encode(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "char_files"
    max_seq_length = 512
    dataset = SenterDataset(str(path), char_tokenizer, max_seq_length)
    num_examples = len(dataset)

    sent_segmentation_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    sent_segmentation_labels[0, 1] = SENT_SEGMENTATION_TAGS.index("B")  # 花
    sent_segmentation_labels[0, 2] = SENT_SEGMENTATION_TAGS.index("I")  # 咲
    sent_segmentation_labels[0, 3] = SENT_SEGMENTATION_TAGS.index("I")  # ガ
    sent_segmentation_labels[0, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ニ
    sent_segmentation_labels[0, 5] = SENT_SEGMENTATION_TAGS.index("I")  # を
    sent_segmentation_labels[0, 6] = SENT_SEGMENTATION_TAGS.index("I")  # 買
    sent_segmentation_labels[0, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ぅ
    sent_segmentation_labels[1, 1] = SENT_SEGMENTATION_TAGS.index("B")  # う
    sent_segmentation_labels[1, 2] = SENT_SEGMENTATION_TAGS.index("I")  # ま
    sent_segmentation_labels[1, 3] = SENT_SEGMENTATION_TAGS.index("I")  # そ
    sent_segmentation_labels[1, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ー
    sent_segmentation_labels[1, 5] = SENT_SEGMENTATION_TAGS.index("I")  # で
    sent_segmentation_labels[1, 6] = SENT_SEGMENTATION_TAGS.index("I")  # す
    sent_segmentation_labels[1, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ね
    sent_segmentation_labels[1, 8] = SENT_SEGMENTATION_TAGS.index("I")  # 〜
    sent_segmentation_labels[1, 9] = SENT_SEGMENTATION_TAGS.index("I")  # 〜
    assert dataset[0].sent_segmentation_labels == sent_segmentation_labels[0].tolist()
    assert dataset[1].sent_segmentation_labels == sent_segmentation_labels[1].tolist()
