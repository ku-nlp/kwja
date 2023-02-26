from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from kwja.datamodule.datasets import TypoDataset
from kwja.metrics import TypoModuleMetric


def test_typo_module_metric() -> None:
    metric = TypoModuleMetric(confidence_thresholds=(0.0, 0.8, 0.9))

    path = Path(__file__).absolute().parent.parent / "data" / "datasets" / "typo_files"
    tokenizer = AutoTokenizer.from_pretrained(
        "ku-nlp/roberta-base-japanese-char-wwm",
        do_word_tokenize=False,
        additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
    )
    max_seq_length = 20
    dataset = TypoDataset(str(path), tokenizer, max_seq_length=max_seq_length)
    metric.set_properties(dataset)

    metric.update(
        {
            "example_ids": torch.empty(0),  # dummy
            "kdr_predictions": torch.empty(0),
            "kdr_probabilities": torch.empty(0),
            "ins_predictions": torch.empty(0),
            "ins_probabilities": torch.empty(0),
        }
    )

    num_examples = len(dataset)
    metric.example_ids = torch.arange(num_examples, dtype=torch.long)

    kdr_probabilities = torch.zeros((num_examples, max_seq_length, len(dataset.token2token_id)), dtype=torch.float)
    kdr_probabilities[0, 1, dataset.token2token_id["松"]] = 0.65  # 待 -> 松
    kdr_probabilities[0, 2, dataset.token2token_id["<d>"]] = 0.85  # つ -> φ
    kdr_probabilities[0, 3, dataset.token2token_id["<k>"]] = 1.0  # の
    kdr_probabilities[0, 4, dataset.token2token_id["<k>"]] = 1.0  # 木
    kdr_probabilities[0, 5, dataset.token2token_id["<k>"]] = 1.0  # が
    kdr_probabilities[0, 6, dataset.token2token_id["<k>"]] = 1.0  # 枯
    kdr_probabilities[0, 7, dataset.token2token_id["<k>"]] = 1.0  # れ
    kdr_probabilities[0, 8, dataset.token2token_id["<k>"]] = 1.0  # る
    kdr_probabilities[1, 1:9, dataset.token2token_id["<k>"]] = 1.0  # 紹介ことなかった
    metric.kdr_probabilities, metric.kdr_predictions = kdr_probabilities.max(dim=2)

    ins_probabilities = torch.zeros((num_examples, max_seq_length, len(dataset.token2token_id)), dtype=torch.float)
    ins_probabilities[0, 1:10, dataset.token2token_id["<_>"]] = 1.0  # 待つの木が枯れる
    ins_probabilities[1, 1, dataset.token2token_id["<_>"]] = 1.0  # 紹
    ins_probabilities[1, 2, dataset.token2token_id["<_>"]] = 1.0  # 介
    ins_probabilities[1, 3, dataset.token2token_id["する"]] = 1.0  # こ -> するこ
    ins_probabilities[1, 4, dataset.token2token_id["<_>"]] = 1.0  # と
    ins_probabilities[1, 5, dataset.token2token_id["が"]] = 1.0  # な -> がな
    ins_probabilities[1, 6, dataset.token2token_id["<_>"]] = 1.0  # か
    ins_probabilities[1, 7, dataset.token2token_id["<_>"]] = 1.0  # っ
    ins_probabilities[1, 8, dataset.token2token_id["<_>"]] = 1.0  # た
    ins_probabilities[1, 9, dataset.token2token_id["<_>"]] = 1.0  # <dummy>
    metric.ins_probabilities, metric.ins_predictions = ins_probabilities.max(dim=2)

    metrics = metric.compute()

    # tp = 5, fp = 0, fn = 0
    assert metrics["typo_correction_0.0_precision"] == pytest.approx(5 / 5)
    assert metrics["typo_correction_0.0_recall"] == pytest.approx(5 / 5)
    assert metrics["typo_correction_0.0_f1"] == pytest.approx((2 * 5 / 5 * 5 / 5) / (5 / 5 + 5 / 5))
    assert metrics["typo_correction_0.0_f0.5"] == pytest.approx((1.25 * 5 / 5 * 5 / 5) / (0.25 * 5 / 5 + 5 / 5))
    # tp = 4, fp = 0, fn = 1
    assert metrics["typo_correction_0.8_precision"] == pytest.approx(4 / 4)
    assert metrics["typo_correction_0.8_recall"] == pytest.approx(4 / 5)
    assert metrics["typo_correction_0.8_f1"] == pytest.approx((2 * 4 / 4 * 4 / 5) / (4 / 4 + 4 / 5))
    assert metrics["typo_correction_0.8_f0.5"] == pytest.approx((1.25 * 4 / 4 * 4 / 5) / (0.25 * 4 / 4 + 4 / 5))
    # tp = 3, fp = 0, fn = 2
    assert metrics["typo_correction_0.9_precision"] == pytest.approx(3 / 3)
    assert metrics["typo_correction_0.9_recall"] == pytest.approx(3 / 5)
    assert metrics["typo_correction_0.9_f1"] == pytest.approx((2 * 3 / 3 * 3 / 5) / (3 / 3 + 3 / 5))
    assert metrics["typo_correction_0.9_f0.5"] == pytest.approx((1.25 * 3 / 3 * 3 / 5) / (0.25 * 3 / 3 + 3 / 5))
