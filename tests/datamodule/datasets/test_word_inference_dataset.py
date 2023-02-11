import tempfile
import textwrap
from pathlib import Path

from omegaconf import ListConfig
from transformers import AutoTokenizer

import kwja
from kwja.datamodule.datasets.word_inference_dataset import WordInferenceDataset

exophora_referents = ["著者", "読者", "不特定:人", "不特定:物"]
special_tokens = exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese", additional_special_tokens=special_tokens)
word_dataset_args = {
    "document_split_stride": 1,
    "cohesion_tasks": ListConfig(["pas_analysis", "bridging_reference_resolution", "coreference_resolution"]),
    "exophora_referents": ListConfig(exophora_referents),
    "restrict_cohesion_target": True,
    "pas_cases": ListConfig(["ガ", "ヲ", "ニ", "ガ２"]),
    "br_cases": ListConfig(["ノ"]),
    "special_tokens": ListConfig(special_tokens),
    "max_seq_length": 128,
}


def test_init():
    _ = WordInferenceDataset(tokenizer, **word_dataset_args)


def test_len():
    juman_texts = [
        textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
    ]

    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write("".join(juman_texts))
    juman_file.seek(0)

    dataset = WordInferenceDataset(tokenizer, **word_dataset_args, juman_file=Path(juman_file.name))
    assert len(dataset) == 1


def test_len_multi_doc():
    juman_texts = [
        textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        ),
        textwrap.dedent(
            f"""\
            # S-ID:test-1-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            雨 _ 雨 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        ),
    ]

    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write("".join(juman_texts))
    juman_file.seek(0)

    dataset = WordInferenceDataset(tokenizer, **word_dataset_args, juman_file=Path(juman_file.name))
    assert len(dataset) == 2


def test_getitem():
    juman_texts = [
        textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
    ]

    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write("".join(juman_texts))
    juman_file.seek(0)

    max_seq_length = 256
    dataset = WordInferenceDataset(
        tokenizer, **{**word_dataset_args, "max_seq_length": max_seq_length}, juman_file=Path(juman_file.name)
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
