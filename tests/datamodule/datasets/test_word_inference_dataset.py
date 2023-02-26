import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List

import pytest
from omegaconf import ListConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets.word_inference_dataset import WordInferenceDataset


@pytest.fixture()
def path() -> Path:
    return Path(__file__).absolute().parent.parent.parent / "data" / "datasets" / "word_files"


@pytest.fixture()
def cohesion_tasks() -> List[str]:
    return ["pas_analysis", "bridging_reference_resolution", "coreference_resolution"]


@pytest.fixture()
def exophora_referents() -> List[str]:
    return ["著者", "読者", "不特定:人", "不特定:物"]


@pytest.fixture()
def pas_cases() -> List[str]:
    return ["ガ", "ヲ", "ニ", "ガ２"]


@pytest.fixture()
def br_cases() -> List[str]:
    return ["ノ"]


@pytest.fixture()
def special_tokens(exophora_referents: List[str]) -> List[str]:
    return exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]


@pytest.fixture()
def tokenizer(special_tokens: List[str]) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese", additional_special_tokens=special_tokens)


@pytest.fixture()
def dataset_kwargs(
    cohesion_tasks: List[str],
    exophora_referents: List[str],
    pas_cases: List[str],
    br_cases: List[str],
    special_tokens: List[str],
) -> Dict[str, Any]:
    return {
        "cohesion_tasks": ListConfig(cohesion_tasks),
        "exophora_referents": ListConfig(exophora_referents),
        "restrict_cohesion_target": True,
        "pas_cases": ListConfig(pas_cases),
        "br_cases": ListConfig(br_cases),
        "special_tokens": ListConfig(special_tokens),
    }


def test_init(tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    _ = WordInferenceDataset(tokenizer, max_seq_length=256, document_split_stride=1, **dataset_kwargs)


def test_len(tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = WordInferenceDataset(
        tokenizer, max_seq_length=256, document_split_stride=1, juman_file=Path(juman_file.name), **dataset_kwargs
    )
    assert len(dataset) == 1


def test_len_multi_doc(tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        雨 _ 雨 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = WordInferenceDataset(
        tokenizer, max_seq_length=256, document_split_stride=1, juman_file=Path(juman_file.name), **dataset_kwargs
    )
    assert len(dataset) == 2


def test_getitem(tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    max_seq_length = 256
    dataset = WordInferenceDataset(
        tokenizer,
        max_seq_length=max_seq_length,
        document_split_stride=1,
        juman_file=Path(juman_file.name),
        **dataset_kwargs,
    )
    num_cohesion_rels = len([r for utils in dataset.cohesion_task2utils.values() for r in utils.rels])
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target_mask" in item
        assert "subword_map" in item
        assert "reading_subword_map" in item
        assert "dependency_mask" in item
        assert "cohesion_mask" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["target_mask"].shape == (max_seq_length,)
        assert item["subword_map"].shape == (max_seq_length, max_seq_length)
        assert item["reading_subword_map"].shape == (max_seq_length, max_seq_length)
        assert item["dependency_mask"].shape == (
            max_seq_length,
            max_seq_length,
        )
        assert item["cohesion_mask"].shape == (
            num_cohesion_rels,
            max_seq_length,
            max_seq_length,
        )
