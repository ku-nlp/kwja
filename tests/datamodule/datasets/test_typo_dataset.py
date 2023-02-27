from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import TypoDataset
from kwja.utils.constants import IGNORE_INDEX


def test_init(fixture_data_dir: Path, typo_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "typo_files"
    _ = TypoDataset(str(path), typo_tokenizer, max_seq_length=256)


def test_getitem(fixture_data_dir: Path, typo_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "typo_files"
    max_seq_length = 256
    dataset = TypoDataset(str(path), typo_tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.kdr_labels) == max_seq_length
        assert len(feature.ins_labels) == max_seq_length

        kdr_labels = [x for x in feature.kdr_labels if x != IGNORE_INDEX]
        ins_labels = [x for x in feature.ins_labels if x != IGNORE_INDEX]
        assert len(dataset.examples[i].pre_text) == len(kdr_labels) == len(ins_labels) - 1


def test_encode(fixture_data_dir: Path, typo_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "typo_files"
    max_seq_length = 256
    dataset = TypoDataset(str(path), typo_tokenizer, max_seq_length)

    kdr_labels = torch.full((len(dataset), max_seq_length), IGNORE_INDEX, dtype=torch.long)
    kdr_labels[0, 1] = dataset.token2token_id["松"]  # 待 -> 松
    kdr_labels[0, 2] = dataset.token2token_id["<d>"]  # つ -> φ
    kdr_labels[0, 3] = dataset.token2token_id["<k>"]  # の
    kdr_labels[0, 4] = dataset.token2token_id["<k>"]  # 木
    kdr_labels[0, 5] = dataset.token2token_id["<k>"]  # が
    kdr_labels[0, 6] = dataset.token2token_id["<k>"]  # 枯
    kdr_labels[0, 7] = dataset.token2token_id["<k>"]  # れ
    kdr_labels[0, 8] = dataset.token2token_id["<k>"]  # る
    kdr_labels[1, 1:9] = dataset.token2token_id["<k>"]  # 紹介ことなかった
    assert dataset[0].kdr_labels == kdr_labels[0].tolist()
    assert dataset[1].kdr_labels == kdr_labels[1].tolist()

    ins_labels = torch.full((len(dataset), max_seq_length), IGNORE_INDEX, dtype=torch.long)
    ins_labels[0, 1:10] = dataset.token2token_id["<_>"]  # 待つの木が枯れる
    ins_labels[1, 1] = dataset.token2token_id["<_>"]  # 紹
    ins_labels[1, 2] = dataset.token2token_id["<_>"]  # 介
    ins_labels[1, 3] = dataset.token2token_id["する"]  # こ -> するこ
    ins_labels[1, 4] = dataset.token2token_id["<_>"]  # と
    ins_labels[1, 5] = dataset.token2token_id["が"]  # な -> がな
    ins_labels[1, 6] = dataset.token2token_id["<_>"]  # か
    ins_labels[1, 7] = dataset.token2token_id["<_>"]  # っ
    ins_labels[1, 8] = dataset.token2token_id["<_>"]  # た
    ins_labels[1, 9] = dataset.token2token_id["<_>"]  # <dummy>
    assert dataset[0].ins_labels == ins_labels[0].tolist()
    assert dataset[1].ins_labels == ins_labels[1].tolist()
