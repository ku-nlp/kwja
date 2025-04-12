from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Optional, Union

import pytest
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.callbacks import TypoModuleWriter
from kwja.datamodule.datasets import TypoInferenceDataset
from kwja.datamodule.datasets.typo import get_maps


class MockTrainer:
    def __init__(self, predict_dataloaders: list[DataLoader]):
        self.predict_dataloaders = predict_dataloaders


@pytest.mark.parametrize(
    "destination",
    [
        None,
        Path(TemporaryDirectory().name) / "typo_prediction.juman",
        str(Path(TemporaryDirectory().name) / "typo_prediction.juman"),
    ],
)
def test_init(destination: Optional[Union[str, Path]], typo_tokenizer: PreTrainedTokenizerBase):
    _ = TypoModuleWriter(confidence_threshold=0.9, tokenizer=typo_tokenizer, destination=destination)


def test_write_on_batch_end(typo_tokenizer: PreTrainedTokenizerBase):
    texts = ["この文は解析されません…", "待つの木が枯れる", "紹介ことなかった", "この文は解析されません…"]
    num_examples = 2  # num_stash = 2
    max_seq_length = 32  # >= 11

    token2token_id, _ = get_maps(typo_tokenizer)

    kdr_probabilities = torch.zeros((num_examples, max_seq_length, len(token2token_id)), dtype=torch.float)
    kdr_probabilities[0, 0, token2token_id["<k>"]] = 1.0  # [CLS]
    kdr_probabilities[0, 1, token2token_id["松"]] = 0.65  # 待 -> 松
    kdr_probabilities[0, 2, token2token_id["<d>"]] = 0.85  # つ -> φ
    kdr_probabilities[0, 3, token2token_id["<k>"]] = 1.0  # の
    kdr_probabilities[0, 4, token2token_id["<k>"]] = 1.0  # 木
    kdr_probabilities[0, 5, token2token_id["<k>"]] = 1.0  # が
    kdr_probabilities[0, 6, token2token_id["<k>"]] = 1.0  # 枯
    kdr_probabilities[0, 7, token2token_id["<k>"]] = 1.0  # れ
    kdr_probabilities[0, 8, token2token_id["<k>"]] = 1.0  # る
    kdr_probabilities[0, 9, token2token_id["<k>"]] = 1.0  # <dummy>
    kdr_probabilities[0, 10, token2token_id["<k>"]] = 1.0  # [SEP]
    kdr_probabilities[1, :, token2token_id["<k>"]] = 1.0

    ins_probabilities = torch.zeros((num_examples, max_seq_length, len(token2token_id)), dtype=torch.float)
    ins_probabilities[0, :, token2token_id["<_>"]] = 1.0
    ins_probabilities[1, 0, token2token_id["<_>"]] = 1.0  # [CLS]
    ins_probabilities[1, 1, token2token_id["<_>"]] = 1.0  # 紹
    ins_probabilities[1, 2, token2token_id["<_>"]] = 1.0  # 介
    ins_probabilities[1, 3, token2token_id["する"]] = 1.0  # こ -> するこ
    ins_probabilities[1, 4, token2token_id["<_>"]] = 1.0  # と
    ins_probabilities[1, 5, token2token_id["が"]] = 1.0  # な -> がな
    ins_probabilities[1, 6, token2token_id["<_>"]] = 1.0  # か
    ins_probabilities[1, 7, token2token_id["<_>"]] = 1.0  # っ
    ins_probabilities[1, 8, token2token_id["<_>"]] = 1.0  # た
    ins_probabilities[1, 9, token2token_id["<_>"]] = 1.0  # <dummy>
    ins_probabilities[1, 10, token2token_id["<_>"]] = 1.0  # [SEP]

    kdr_max_probabilities, kdr_predictions = kdr_probabilities.max(dim=2)
    ins_max_probabilities, ins_predictions = ins_probabilities.max(dim=2)
    prediction = {
        "example_ids": torch.arange(num_examples, dtype=torch.long),
        "kdr_predictions": kdr_predictions,
        "kdr_probabilities": kdr_max_probabilities,
        "ins_predictions": ins_predictions,
        "ins_probabilities": ins_max_probabilities,
    }

    with TemporaryDirectory() as tmp_dir:
        destination = tmp_dir / Path("typo_prediction.txt")
        confidence_thresholds = [0.0, 0.8, 0.9]
        expected_texts = [
            dedent(
                """\
                この文は解析されません…
                EOD
                松の木が枯れる
                EOD
                紹介することがなかった
                EOD
                この文は解析されません…
                EOD
                """
            ),
            dedent(
                """\
                この文は解析されません…
                EOD
                待の木が枯れる
                EOD
                紹介することがなかった
                EOD
                この文は解析されません…
                EOD
                """
            ),
            dedent(
                """\
                この文は解析されません…
                EOD
                待つの木が枯れる
                EOD
                紹介することがなかった
                EOD
                この文は解析されません…
                EOD
                """
            ),
        ]
        for confidence_threshold, expected_text in zip(confidence_thresholds, expected_texts):
            dataset = TypoInferenceDataset(ListConfig(texts), typo_tokenizer, max_seq_length)
            trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])
            writer = TypoModuleWriter(confidence_threshold, typo_tokenizer, destination=destination)
            writer.write_on_batch_end(trainer, ..., prediction, None, ..., 0, 0)  # type: ignore
            assert isinstance(writer.destination, Path), "destination isn't set"
            assert writer.destination.read_text() == expected_text
