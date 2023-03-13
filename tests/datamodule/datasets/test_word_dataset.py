from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from rhoknp import Document
from rhoknp.props import DepType
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordDataset
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    NE_TAGS,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
)


@pytest.fixture()
def split_into_words_word_tokenizer(special_tokens: List[str]) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese", additional_special_tokens=special_tokens)


def test_init(fixture_data_dir: Path, word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    path = fixture_data_dir / "datasets" / "word_files"
    _ = WordDataset(str(path), word_tokenizer, max_seq_length=256, document_split_stride=1, **dataset_kwargs)


def test_getitem(fixture_data_dir: Path, word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    path = fixture_data_dir / "datasets" / "word_files"
    max_seq_length = 256
    dataset = WordDataset(str(path), word_tokenizer, max_seq_length, document_split_stride=1, **dataset_kwargs)
    num_cohesion_rels = len([r for utils in dataset.cohesion_task2utils.values() for r in utils.rels])
    for i in range(len(dataset)):
        document = dataset.documents[i]
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.target_mask) == max_seq_length
        assert np.array(feature.subword_map).shape == (max_seq_length, max_seq_length)
        assert (np.array(feature.subword_map).sum(axis=1) != 0).sum() == len(
            document.morphemes
        ) + dataset.num_special_tokens
        assert len(feature.reading_labels) == max_seq_length
        assert np.array(feature.reading_subword_map).shape == (max_seq_length, max_seq_length)
        assert (np.array(feature.reading_subword_map).sum(axis=1) != 0).sum() == len(document.morphemes)
        assert len(feature.pos_labels) == max_seq_length
        assert len(feature.subpos_labels) == max_seq_length
        assert len(feature.conjtype_labels) == max_seq_length
        assert len(feature.conjform_labels) == max_seq_length
        assert np.array(feature.word_feature_labels).shape == (max_seq_length, len(WORD_FEATURES))
        assert len(feature.ne_labels) == max_seq_length
        assert np.array(feature.base_phrase_feature_labels).shape == (max_seq_length, len(BASE_PHRASE_FEATURES))
        assert len(feature.dependency_labels) == max_seq_length
        assert np.array(feature.dependency_mask).shape == (max_seq_length, max_seq_length)
        assert len(feature.dependency_type_labels) == max_seq_length
        assert np.array(feature.cohesion_labels).shape == (num_cohesion_rels, max_seq_length, max_seq_length)
        assert np.array(feature.cohesion_mask).shape == (num_cohesion_rels, max_seq_length, max_seq_length)
        assert np.array(feature.discourse_labels).shape == (max_seq_length, max_seq_length)


def test_encode(fixture_data_dir: Path, word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]):
    path = fixture_data_dir / "datasets" / "word_files"
    max_seq_length = 32
    dataset = WordDataset(
        str(path), word_tokenizer, max_seq_length=max_seq_length, document_split_stride=1, **dataset_kwargs
    )
    assert dataset.tokenizer_input_format == "text"
    dataset.examples[1].load_discourse_document(Document.from_knp(path.joinpath("1.knp").read_text()))
    num_examples = len(dataset)

    reading_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    reading_labels[0, 1] = dataset.reading2reading_id["たろう"]  # 太郎 -> たろう
    reading_labels[0, 2] = dataset.reading2reading_id["[ID]"]  # と
    reading_labels[0, 3] = dataset.reading2reading_id["じろう"]  # 次郎 -> じろう
    reading_labels[0, 4] = dataset.reading2reading_id["[ID]"]  # は
    reading_labels[0, 5] = dataset.reading2reading_id["[ID]"]  # よく
    reading_labels[0, 6] = dataset.reading2reading_id["[ID]"]  # けん
    reading_labels[0, 7] = dataset.reading2reading_id["[ID]"]  # か
    reading_labels[0, 8] = dataset.reading2reading_id["[ID]"]  # する
    reading_labels[1, 1] = dataset.reading2reading_id["からい"]  # 辛い -> からい
    reading_labels[1, 2] = dataset.reading2reading_id["らーめん"]  # ラーメン -> らーめん
    reading_labels[1, 3] = dataset.reading2reading_id["[ID]"]  # が
    reading_labels[1, 4] = dataset.reading2reading_id["すきな"]  # 好きな -> すきな
    reading_labels[1, 5] = dataset.reading2reading_id["[ID]"]  # ので
    reading_labels[1, 6] = dataset.reading2reading_id["たのみ"]  # 頼み -> たのみ
    reading_labels[1, 7] = dataset.reading2reading_id["[ID]"]  # ました
    assert reading_labels[0].tolist() == dataset[0].reading_labels
    assert reading_labels[1].tolist() == dataset[1].reading_labels

    reading_subword_map = torch.zeros((num_examples, max_seq_length, max_seq_length), dtype=torch.bool)
    reading_subword_map[0, 0, 1] = True
    reading_subword_map[0, 1, 2] = True
    reading_subword_map[0, 2, 3] = True
    reading_subword_map[0, 3, 4] = True
    reading_subword_map[0, 4, 5] = True
    reading_subword_map[0, 5, 6] = True  # けんか -> けん
    reading_subword_map[0, 5, 7] = True  # けんか -> か
    reading_subword_map[0, 6, 8] = True
    reading_subword_map[1, 0, 1] = True
    reading_subword_map[1, 1, 2] = True
    reading_subword_map[1, 2, 3] = True
    reading_subword_map[1, 3, 4] = True
    reading_subword_map[1, 4, 5] = True
    reading_subword_map[1, 5, 6] = True
    reading_subword_map[1, 6, 7] = True
    assert reading_subword_map[0].tolist() == dataset[0].reading_subword_map
    assert reading_subword_map[1].tolist() == dataset[1].reading_subword_map

    pos_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    pos_labels[0, 0] = POS_TAGS.index("名詞")  # 太郎
    pos_labels[0, 1] = POS_TAGS.index("助詞")  # と
    pos_labels[0, 2] = POS_TAGS.index("名詞")  # 次郎
    pos_labels[0, 3] = POS_TAGS.index("助詞")  # は
    pos_labels[0, 4] = POS_TAGS.index("副詞")  # よく
    pos_labels[0, 5] = POS_TAGS.index("名詞")  # けんか
    pos_labels[0, 6] = POS_TAGS.index("動詞")  # する
    pos_labels[1, 0] = POS_TAGS.index("形容詞")  # 辛い
    pos_labels[1, 1] = POS_TAGS.index("名詞")  # ラーメン
    pos_labels[1, 2] = POS_TAGS.index("助詞")  # が
    pos_labels[1, 3] = POS_TAGS.index("形容詞")  # 好きな
    pos_labels[1, 4] = POS_TAGS.index("助動詞")  # ので
    pos_labels[1, 5] = POS_TAGS.index("動詞")  # 頼み
    pos_labels[1, 6] = POS_TAGS.index("接尾辞")  # ました
    assert pos_labels[0].tolist() == dataset[0].pos_labels
    assert pos_labels[1].tolist() == dataset[1].pos_labels

    subpos_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    subpos_labels[0, 0] = SUBPOS_TAGS.index("人名")  # 太郎
    subpos_labels[0, 1] = SUBPOS_TAGS.index("格助詞")  # と
    subpos_labels[0, 2] = SUBPOS_TAGS.index("人名")  # 次郎
    subpos_labels[0, 3] = SUBPOS_TAGS.index("副助詞")  # は
    subpos_labels[0, 4] = SUBPOS_TAGS.index("*")  # よく
    subpos_labels[0, 5] = SUBPOS_TAGS.index("サ変名詞")  # けんか
    subpos_labels[0, 6] = SUBPOS_TAGS.index("*")  # する
    subpos_labels[1, 0] = SUBPOS_TAGS.index("*")  # 辛い
    subpos_labels[1, 1] = SUBPOS_TAGS.index("普通名詞")  # ラーメン
    subpos_labels[1, 2] = SUBPOS_TAGS.index("格助詞")  # が
    subpos_labels[1, 3] = SUBPOS_TAGS.index("*")  # 好きな
    subpos_labels[1, 4] = SUBPOS_TAGS.index("*")  # ので
    subpos_labels[1, 5] = SUBPOS_TAGS.index("*")  # 頼み
    subpos_labels[1, 6] = SUBPOS_TAGS.index("動詞性接尾辞")  # ました
    assert subpos_labels[0].tolist() == dataset[0].subpos_labels
    assert subpos_labels[1].tolist() == dataset[1].subpos_labels

    conjtype_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    conjtype_labels[0, 0] = CONJTYPE_TAGS.index("*")  # 太郎
    conjtype_labels[0, 1] = CONJTYPE_TAGS.index("*")  # と
    conjtype_labels[0, 2] = CONJTYPE_TAGS.index("*")  # 次郎
    conjtype_labels[0, 3] = CONJTYPE_TAGS.index("*")  # は
    conjtype_labels[0, 4] = CONJTYPE_TAGS.index("*")  # よく
    conjtype_labels[0, 5] = CONJTYPE_TAGS.index("*")  # けんか
    conjtype_labels[0, 6] = CONJTYPE_TAGS.index("サ変動詞")  # する
    conjtype_labels[1, 0] = CONJTYPE_TAGS.index("イ形容詞アウオ段")  # 辛い
    conjtype_labels[1, 1] = CONJTYPE_TAGS.index("*")  # ラーメン
    conjtype_labels[1, 2] = CONJTYPE_TAGS.index("*")  # が
    conjtype_labels[1, 3] = CONJTYPE_TAGS.index("ナ形容詞")  # 好きな
    conjtype_labels[1, 4] = CONJTYPE_TAGS.index("ナ形容詞")  # ので
    conjtype_labels[1, 5] = CONJTYPE_TAGS.index("子音動詞マ行")  # 頼み
    conjtype_labels[1, 6] = CONJTYPE_TAGS.index("動詞性接尾辞ます型")  # ました
    assert conjtype_labels[0].tolist() == dataset[0].conjtype_labels
    assert conjtype_labels[1].tolist() == dataset[1].conjtype_labels

    conjform_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    conjform_labels[0, 0] = CONJFORM_TAGS.index("*")  # 太郎
    conjform_labels[0, 1] = CONJFORM_TAGS.index("*")  # と
    conjform_labels[0, 2] = CONJFORM_TAGS.index("*")  # 次郎
    conjform_labels[0, 3] = CONJFORM_TAGS.index("*")  # は
    conjform_labels[0, 4] = CONJFORM_TAGS.index("*")  # よく
    conjform_labels[0, 5] = CONJFORM_TAGS.index("*")  # けんか
    conjform_labels[0, 6] = CONJFORM_TAGS.index("基本形")  # する
    conjform_labels[1, 0] = CONJFORM_TAGS.index("基本形")  # 辛い
    conjform_labels[1, 1] = CONJFORM_TAGS.index("*")  # ラーメン
    conjform_labels[1, 2] = CONJFORM_TAGS.index("*")  # が
    conjform_labels[1, 3] = CONJFORM_TAGS.index("ダ列基本連体形")  # 好きな
    conjform_labels[1, 4] = CONJFORM_TAGS.index("ダ列タ系連用テ形")  # ので
    conjform_labels[1, 5] = CONJFORM_TAGS.index("基本連用形")  # 頼み
    conjform_labels[1, 6] = CONJFORM_TAGS.index("タ形")  # ました
    assert conjform_labels[0].tolist() == dataset[0].conjform_labels
    assert conjform_labels[1].tolist() == dataset[1].conjform_labels

    word_feature_labels = torch.zeros((num_examples, max_seq_length, len(WORD_FEATURES)), dtype=torch.long)
    word_feature_labels[0, 0, WORD_FEATURES.index("基本句-主辞")] = 1  # 太郎
    word_feature_labels[0, 1, WORD_FEATURES.index("基本句-区切")] = 1  # と
    word_feature_labels[0, 1, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 2, WORD_FEATURES.index("基本句-主辞")] = 1  # 次郎
    word_feature_labels[0, 3, WORD_FEATURES.index("基本句-区切")] = 1  # は
    word_feature_labels[0, 3, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("基本句-主辞")] = 1  # よく
    word_feature_labels[0, 4, WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[0, 5, WORD_FEATURES.index("基本句-主辞")] = 1  # けんか
    word_feature_labels[0, 5, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[0, 5, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[0, 6, WORD_FEATURES.index("基本句-区切")] = 1  # する
    word_feature_labels[0, 6, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 7:, :] = IGNORE_INDEX
    word_feature_labels[1, 0, WORD_FEATURES.index("基本句-主辞")] = 1  # 辛い
    word_feature_labels[1, 0, WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 1, WORD_FEATURES.index("基本句-主辞")] = 1  # ラーメン
    word_feature_labels[1, 2, WORD_FEATURES.index("基本句-区切")] = 1  # が
    word_feature_labels[1, 2, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 3, WORD_FEATURES.index("基本句-主辞")] = 1  # 好きな
    word_feature_labels[1, 3, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 3, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 4, WORD_FEATURES.index("基本句-区切")] = 1  # ので
    word_feature_labels[1, 4, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 5, WORD_FEATURES.index("基本句-主辞")] = 1  # 頼み
    word_feature_labels[1, 5, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 5, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 6, WORD_FEATURES.index("基本句-区切")] = 1  # ました
    word_feature_labels[1, 6, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 7:, :] = IGNORE_INDEX
    assert word_feature_labels[0].tolist() == dataset[0].word_feature_labels
    assert word_feature_labels[1].tolist() == dataset[1].word_feature_labels

    ne_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    ne_labels[0, 0] = NE_TAGS.index("B-PERSON")  # 太郎
    ne_labels[0, 1] = NE_TAGS.index("O")  # と
    ne_labels[0, 2] = NE_TAGS.index("B-PERSON")  # 次郎
    ne_labels[0, 3:7] = NE_TAGS.index("O")  # は よく けんか する
    ne_labels[1, :7] = NE_TAGS.index("O")  # 辛い ラーメン が 好きな ので 頼み ました
    assert ne_labels[0].tolist() == dataset[0].ne_labels
    assert ne_labels[1].tolist() == dataset[1].ne_labels

    base_phrase_feature_labels = torch.zeros(
        (num_examples, max_seq_length, len(BASE_PHRASE_FEATURES)), dtype=torch.long
    )
    base_phrase_feature_labels[0, 0, BASE_PHRASE_FEATURES.index("体言")] = 1  # 太郎
    base_phrase_feature_labels[0, 0, BASE_PHRASE_FEATURES.index("SM-主体")] = 1
    base_phrase_feature_labels[0, 1, :] = IGNORE_INDEX  # と
    base_phrase_feature_labels[0, 2, BASE_PHRASE_FEATURES.index("体言")] = 1  # 次郎
    base_phrase_feature_labels[0, 2, BASE_PHRASE_FEATURES.index("SM-主体")] = 1
    base_phrase_feature_labels[0, 3, :] = IGNORE_INDEX  # は
    base_phrase_feature_labels[0, 4, BASE_PHRASE_FEATURES.index("修飾")] = 1  # よく
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1  # けんか
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1
    base_phrase_feature_labels[0, 6:, :] = IGNORE_INDEX  # する 〜
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("用言:形")] = 1  # 辛い
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("レベル:B-")] = 1
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("状態述語")] = 1
    base_phrase_feature_labels[1, 1, BASE_PHRASE_FEATURES.index("体言")] = 1  # ラーメン
    base_phrase_feature_labels[1, 2, :] = IGNORE_INDEX  # が
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("用言:形")] = 1  # 好きな
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("レベル:B+")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("状態述語")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-機能-原因・理由")] = 1
    base_phrase_feature_labels[1, 4, :] = IGNORE_INDEX  # ので
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1  # 頼み
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("時制:過去")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("敬語:丁寧表現")] = 1
    base_phrase_feature_labels[1, 6:, :] = IGNORE_INDEX  # ました 〜
    assert base_phrase_feature_labels[0].tolist() == dataset[0].base_phrase_feature_labels
    assert base_phrase_feature_labels[1].tolist() == dataset[1].base_phrase_feature_labels

    dependency_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    dependency_labels[0, 0] = 2  # 太郎 -> 次郎
    dependency_labels[0, 1] = 0  # と -> 太郎
    dependency_labels[0, 2] = 5  # 次郎 -> けんか
    dependency_labels[0, 3] = 2  # は -> 次郎
    dependency_labels[0, 4] = 5  # よく -> けんか
    dependency_labels[0, 5] = dataset.special_token2index["[ROOT]"]  # けんか -> [ROOT]
    dependency_labels[0, 6] = 5  # する -> けんか
    dependency_labels[1, 0] = 1  # 辛い -> ラーメン
    dependency_labels[1, 1] = 3  # ラーメン -> 好きな
    dependency_labels[1, 2] = 1  # が -> ラーメン
    dependency_labels[1, 3] = 5  # 好きな -> 頼み
    dependency_labels[1, 4] = 3  # ので -> 好きな
    dependency_labels[1, 5] = dataset.special_token2index["[ROOT]"]  # 頼み -> [ROOT]
    dependency_labels[1, 6] = 5  # ました -> 頼み
    assert dependency_labels[0].tolist() == dataset[0].dependency_labels
    assert dependency_labels[1].tolist() == dataset[1].dependency_labels

    dependency_type_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    dependency_type_labels[0, 0] = DEPENDENCY_TYPES.index(DepType.PARALLEL)  # 太郎 -> 次郎
    dependency_type_labels[0, 1] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # が -> 太郎
    dependency_type_labels[0, 2] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 次郎 -> けんか
    dependency_type_labels[0, 3] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # は -> 次郎
    dependency_type_labels[0, 4] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # よく -> けんか
    dependency_type_labels[0, 5] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # けんか -> [ROOT]
    dependency_type_labels[0, 6] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # する -> けんか
    dependency_type_labels[1, 0] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 辛い -> ラーメン
    dependency_type_labels[1, 1] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ラーメン -> 好きな
    dependency_type_labels[1, 2] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # が -> ラーメン
    dependency_type_labels[1, 3] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 好きな -> 頼み
    dependency_type_labels[1, 4] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ので -> 好きな
    dependency_type_labels[1, 5] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 頼み -> [ROOT]
    dependency_type_labels[1, 6] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ました -> 頼み
    assert dependency_type_labels[0].tolist() == dataset[0].dependency_type_labels
    assert dependency_type_labels[1].tolist() == dataset[1].dependency_type_labels

    flatten_rels = [r for cohesion_utils in dataset.cohesion_task2utils.values() for r in cohesion_utils.rels]
    cohesion_labels = torch.zeros((num_examples, len(flatten_rels), max_seq_length, max_seq_length), dtype=torch.long)
    cohesion_labels[0, flatten_rels.index("ノ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ノ 太郎
    cohesion_labels[0, flatten_rels.index("="), 0, dataset.special_token2index["[NA]"]] = 1  # 太郎
    cohesion_labels[0, flatten_rels.index("ノ"), 2, dataset.special_token2index["[NULL]"]] = 1  # φ ノ 次郎
    cohesion_labels[0, flatten_rels.index("="), 2, dataset.special_token2index["[NA]"]] = 1  # 次郎
    cohesion_labels[0, flatten_rels.index("ガ"), 5, 2] = 1  # 次郎 ガ けんか
    cohesion_labels[0, flatten_rels.index("ヲ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ けんか
    cohesion_labels[0, flatten_rels.index("ニ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ニ けんか
    cohesion_labels[0, flatten_rels.index("ガ２"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ けんか
    cohesion_labels[1, flatten_rels.index("ガ"), 0, 1] = 1  # ラーメン ガ 辛い
    cohesion_labels[1, flatten_rels.index("ヲ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ 辛い
    cohesion_labels[1, flatten_rels.index("ニ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 辛い
    cohesion_labels[1, flatten_rels.index("ガ２"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ 辛い
    cohesion_labels[1, flatten_rels.index("ノ"), 1, dataset.special_token2index["[NULL]"]] = 1  # φ ノ ラーメン
    cohesion_labels[1, flatten_rels.index("="), 1, dataset.special_token2index["[NA]"]] = 1  # ラーメン
    cohesion_labels[1, flatten_rels.index("ガ"), 3, 1] = 1  # ラーメン ガ 好き
    cohesion_labels[1, flatten_rels.index("ヲ"), 3, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ 好き
    cohesion_labels[1, flatten_rels.index("ニ"), 3, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 好き
    cohesion_labels[1, flatten_rels.index("ガ２"), 3, dataset.special_token2index["著者"]] = 1  # 著者 ガ２ 好き
    cohesion_labels[1, flatten_rels.index("ガ"), 5, dataset.special_token2index["著者"]] = 1  # 著者 ガ 頼み
    cohesion_labels[1, flatten_rels.index("ヲ"), 5, 1] = 1  # ラーメン ヲ 頼み
    cohesion_labels[1, flatten_rels.index("ニ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 頼み
    cohesion_labels[1, flatten_rels.index("ガ２"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ 頼み
    assert cohesion_labels[0].tolist() == dataset[0].cohesion_labels
    assert cohesion_labels[1].tolist() == dataset[1].cohesion_labels

    discourse_labels = torch.full((num_examples, max_seq_length, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    discourse_labels[1, 3, 3] = DISCOURSE_RELATIONS.index("談話関係なし")  # 好きな - 好きな|談話関係なし
    discourse_labels[1, 3, 5] = DISCOURSE_RELATIONS.index("原因・理由")  # 好きな - 頼み|原因・理由
    discourse_labels[1, 5, 3] = DISCOURSE_RELATIONS.index("談話関係なし")  # 頼み - 好きな|談話関係なし
    discourse_labels[1, 5, 5] = DISCOURSE_RELATIONS.index("談話関係なし")  # 頼み - 頼み|談話関係なし
    assert discourse_labels[0].tolist() == dataset[0].discourse_labels
    assert discourse_labels[1].tolist() == dataset[1].discourse_labels


def test_split_into_words_encode(
    fixture_data_dir: Path, split_into_words_word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: Dict[str, Any]
):
    path = fixture_data_dir / "datasets" / "word_files"
    max_seq_length = 32
    dataset = WordDataset(
        str(path),
        split_into_words_word_tokenizer,
        max_seq_length,
        document_split_stride=1,
        **dataset_kwargs,
    )
    assert dataset.tokenizer_input_format == "words"
    dataset.examples[1].load_discourse_document(Document.from_knp(path.joinpath("1.knp").read_text()))
    num_examples = len(dataset)

    reading_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    reading_labels[0, 1] = dataset.reading2reading_id["たろう"]  # 太郎 -> たろう
    reading_labels[0, 2] = dataset.reading2reading_id["[ID]"]  # と
    reading_labels[0, 3] = dataset.reading2reading_id["じろう"]  # 次郎 -> じろう
    reading_labels[0, 4] = dataset.reading2reading_id["[ID]"]  # は
    reading_labels[0, 5] = dataset.reading2reading_id["[ID]"]  # よく
    reading_labels[0, 6] = dataset.reading2reading_id["[ID]"]  # けん
    reading_labels[0, 7] = dataset.reading2reading_id["[ID]"]  # か
    reading_labels[0, 8] = dataset.reading2reading_id["[ID]"]  # する
    reading_labels[1, 1] = dataset.reading2reading_id["からい"]  # 辛い -> からい
    reading_labels[1, 2] = dataset.reading2reading_id["らーめん"]  # ラーメン -> らーめん
    reading_labels[1, 3] = dataset.reading2reading_id["[ID]"]  # が
    reading_labels[1, 4] = dataset.reading2reading_id["すきな"]  # 好きな -> すきな
    reading_labels[1, 5] = dataset.reading2reading_id["[ID]"]  # ので
    reading_labels[1, 6] = dataset.reading2reading_id["たのみ"]  # 頼み -> たのみ
    reading_labels[1, 7] = dataset.reading2reading_id["[ID]"]  # ました
    assert reading_labels[0].tolist() == dataset[0].reading_labels
    assert reading_labels[1].tolist() == dataset[1].reading_labels

    reading_subword_map = torch.zeros((num_examples, max_seq_length, max_seq_length), dtype=torch.bool)
    reading_subword_map[0, 0, 1] = True
    reading_subword_map[0, 1, 2] = True
    reading_subword_map[0, 2, 3] = True
    reading_subword_map[0, 3, 4] = True
    reading_subword_map[0, 4, 5] = True
    reading_subword_map[0, 5, 6] = True  # けんか -> けん
    reading_subword_map[0, 5, 7] = True  # けんか -> か
    reading_subword_map[0, 6, 8] = True
    reading_subword_map[1, 0, 1] = True
    reading_subword_map[1, 1, 2] = True
    reading_subword_map[1, 2, 3] = True
    reading_subword_map[1, 3, 4] = True
    reading_subword_map[1, 4, 5] = True
    reading_subword_map[1, 5, 6] = True
    reading_subword_map[1, 6, 7] = True
    assert reading_subword_map[0].tolist() == dataset[0].reading_subword_map
    assert reading_subword_map[1].tolist() == dataset[1].reading_subword_map

    pos_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    pos_labels[0, 0] = POS_TAGS.index("名詞")  # 太郎
    pos_labels[0, 1] = POS_TAGS.index("助詞")  # と
    pos_labels[0, 2] = POS_TAGS.index("名詞")  # 次郎
    pos_labels[0, 3] = POS_TAGS.index("助詞")  # は
    pos_labels[0, 4] = POS_TAGS.index("副詞")  # よく
    pos_labels[0, 5] = POS_TAGS.index("名詞")  # けんか
    pos_labels[0, 6] = POS_TAGS.index("動詞")  # する
    pos_labels[1, 0] = POS_TAGS.index("形容詞")  # 辛い
    pos_labels[1, 1] = POS_TAGS.index("名詞")  # ラーメン
    pos_labels[1, 2] = POS_TAGS.index("助詞")  # が
    pos_labels[1, 3] = POS_TAGS.index("形容詞")  # 好きな
    pos_labels[1, 4] = POS_TAGS.index("助動詞")  # ので
    pos_labels[1, 5] = POS_TAGS.index("動詞")  # 頼み
    pos_labels[1, 6] = POS_TAGS.index("接尾辞")  # ました
    assert pos_labels[0].tolist() == dataset[0].pos_labels
    assert pos_labels[1].tolist() == dataset[1].pos_labels

    subpos_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    subpos_labels[0, 0] = SUBPOS_TAGS.index("人名")  # 太郎
    subpos_labels[0, 1] = SUBPOS_TAGS.index("格助詞")  # と
    subpos_labels[0, 2] = SUBPOS_TAGS.index("人名")  # 次郎
    subpos_labels[0, 3] = SUBPOS_TAGS.index("副助詞")  # は
    subpos_labels[0, 4] = SUBPOS_TAGS.index("*")  # よく
    subpos_labels[0, 5] = SUBPOS_TAGS.index("サ変名詞")  # けんか
    subpos_labels[0, 6] = SUBPOS_TAGS.index("*")  # する
    subpos_labels[1, 0] = SUBPOS_TAGS.index("*")  # 辛い
    subpos_labels[1, 1] = SUBPOS_TAGS.index("普通名詞")  # ラーメン
    subpos_labels[1, 2] = SUBPOS_TAGS.index("格助詞")  # が
    subpos_labels[1, 3] = SUBPOS_TAGS.index("*")  # 好きな
    subpos_labels[1, 4] = SUBPOS_TAGS.index("*")  # ので
    subpos_labels[1, 5] = SUBPOS_TAGS.index("*")  # 頼み
    subpos_labels[1, 6] = SUBPOS_TAGS.index("動詞性接尾辞")  # ました
    assert subpos_labels[0].tolist() == dataset[0].subpos_labels
    assert subpos_labels[1].tolist() == dataset[1].subpos_labels

    conjtype_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    conjtype_labels[0, 0] = CONJTYPE_TAGS.index("*")  # 太郎
    conjtype_labels[0, 1] = CONJTYPE_TAGS.index("*")  # と
    conjtype_labels[0, 2] = CONJTYPE_TAGS.index("*")  # 次郎
    conjtype_labels[0, 3] = CONJTYPE_TAGS.index("*")  # は
    conjtype_labels[0, 4] = CONJTYPE_TAGS.index("*")  # よく
    conjtype_labels[0, 5] = CONJTYPE_TAGS.index("*")  # けんか
    conjtype_labels[0, 6] = CONJTYPE_TAGS.index("サ変動詞")  # する
    conjtype_labels[1, 0] = CONJTYPE_TAGS.index("イ形容詞アウオ段")  # 辛い
    conjtype_labels[1, 1] = CONJTYPE_TAGS.index("*")  # ラーメン
    conjtype_labels[1, 2] = CONJTYPE_TAGS.index("*")  # が
    conjtype_labels[1, 3] = CONJTYPE_TAGS.index("ナ形容詞")  # 好きな
    conjtype_labels[1, 4] = CONJTYPE_TAGS.index("ナ形容詞")  # ので
    conjtype_labels[1, 5] = CONJTYPE_TAGS.index("子音動詞マ行")  # 頼み
    conjtype_labels[1, 6] = CONJTYPE_TAGS.index("動詞性接尾辞ます型")  # ました
    assert conjtype_labels[0].tolist() == dataset[0].conjtype_labels
    assert conjtype_labels[1].tolist() == dataset[1].conjtype_labels

    conjform_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    conjform_labels[0, 0] = CONJFORM_TAGS.index("*")  # 太郎
    conjform_labels[0, 1] = CONJFORM_TAGS.index("*")  # と
    conjform_labels[0, 2] = CONJFORM_TAGS.index("*")  # 次郎
    conjform_labels[0, 3] = CONJFORM_TAGS.index("*")  # は
    conjform_labels[0, 4] = CONJFORM_TAGS.index("*")  # よく
    conjform_labels[0, 5] = CONJFORM_TAGS.index("*")  # けんか
    conjform_labels[0, 6] = CONJFORM_TAGS.index("基本形")  # する
    conjform_labels[1, 0] = CONJFORM_TAGS.index("基本形")  # 辛い
    conjform_labels[1, 1] = CONJFORM_TAGS.index("*")  # ラーメン
    conjform_labels[1, 2] = CONJFORM_TAGS.index("*")  # が
    conjform_labels[1, 3] = CONJFORM_TAGS.index("ダ列基本連体形")  # 好きな
    conjform_labels[1, 4] = CONJFORM_TAGS.index("ダ列タ系連用テ形")  # ので
    conjform_labels[1, 5] = CONJFORM_TAGS.index("基本連用形")  # 頼み
    conjform_labels[1, 6] = CONJFORM_TAGS.index("タ形")  # ました
    assert conjform_labels[0].tolist() == dataset[0].conjform_labels
    assert conjform_labels[1].tolist() == dataset[1].conjform_labels

    word_feature_labels = torch.zeros((num_examples, max_seq_length, len(WORD_FEATURES)), dtype=torch.long)
    word_feature_labels[0, 0, WORD_FEATURES.index("基本句-主辞")] = 1  # 太郎
    word_feature_labels[0, 1, WORD_FEATURES.index("基本句-区切")] = 1  # と
    word_feature_labels[0, 1, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 2, WORD_FEATURES.index("基本句-主辞")] = 1  # 次郎
    word_feature_labels[0, 3, WORD_FEATURES.index("基本句-区切")] = 1  # は
    word_feature_labels[0, 3, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("基本句-主辞")] = 1  # よく
    word_feature_labels[0, 4, WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[0, 4, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[0, 5, WORD_FEATURES.index("基本句-主辞")] = 1  # けんか
    word_feature_labels[0, 5, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[0, 5, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[0, 6, WORD_FEATURES.index("基本句-区切")] = 1  # する
    word_feature_labels[0, 6, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[0, 7:, :] = IGNORE_INDEX
    word_feature_labels[1, 0, WORD_FEATURES.index("基本句-主辞")] = 1  # 辛い
    word_feature_labels[1, 0, WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 0, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 1, WORD_FEATURES.index("基本句-主辞")] = 1  # ラーメン
    word_feature_labels[1, 2, WORD_FEATURES.index("基本句-区切")] = 1  # が
    word_feature_labels[1, 2, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 3, WORD_FEATURES.index("基本句-主辞")] = 1  # 好きな
    word_feature_labels[1, 3, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 3, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 4, WORD_FEATURES.index("基本句-区切")] = 1  # ので
    word_feature_labels[1, 4, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 5, WORD_FEATURES.index("基本句-主辞")] = 1  # 頼み
    word_feature_labels[1, 5, WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[1, 5, WORD_FEATURES.index("用言表記末尾")] = 1
    word_feature_labels[1, 6, WORD_FEATURES.index("基本句-区切")] = 1  # ました
    word_feature_labels[1, 6, WORD_FEATURES.index("文節-区切")] = 1
    word_feature_labels[1, 7:, :] = IGNORE_INDEX
    assert word_feature_labels[0].tolist() == dataset[0].word_feature_labels
    assert word_feature_labels[1].tolist() == dataset[1].word_feature_labels

    ne_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    ne_labels[0, 0] = NE_TAGS.index("B-PERSON")  # 太郎
    ne_labels[0, 1] = NE_TAGS.index("O")  # と
    ne_labels[0, 2] = NE_TAGS.index("B-PERSON")  # 次郎
    ne_labels[0, 3:7] = NE_TAGS.index("O")  # は よく けんか する
    ne_labels[1, :7] = NE_TAGS.index("O")  # 辛い ラーメン が 好きな ので 頼み ました
    assert ne_labels[0].tolist() == dataset[0].ne_labels
    assert ne_labels[1].tolist() == dataset[1].ne_labels

    base_phrase_feature_labels = torch.zeros(
        (num_examples, max_seq_length, len(BASE_PHRASE_FEATURES)), dtype=torch.long
    )
    base_phrase_feature_labels[0, 0, BASE_PHRASE_FEATURES.index("体言")] = 1  # 太郎
    base_phrase_feature_labels[0, 0, BASE_PHRASE_FEATURES.index("SM-主体")] = 1
    base_phrase_feature_labels[0, 1, :] = IGNORE_INDEX  # と
    base_phrase_feature_labels[0, 2, BASE_PHRASE_FEATURES.index("体言")] = 1  # 次郎
    base_phrase_feature_labels[0, 2, BASE_PHRASE_FEATURES.index("SM-主体")] = 1
    base_phrase_feature_labels[0, 3, :] = IGNORE_INDEX  # は
    base_phrase_feature_labels[0, 4, BASE_PHRASE_FEATURES.index("修飾")] = 1  # よく
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1  # けんか
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[0, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1
    base_phrase_feature_labels[0, 6:, :] = IGNORE_INDEX  # する 〜
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("用言:形")] = 1  # 辛い
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("レベル:B-")] = 1
    base_phrase_feature_labels[1, 0, BASE_PHRASE_FEATURES.index("状態述語")] = 1
    base_phrase_feature_labels[1, 1, BASE_PHRASE_FEATURES.index("体言")] = 1  # ラーメン
    base_phrase_feature_labels[1, 2, :] = IGNORE_INDEX  # が
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("用言:形")] = 1  # 好きな
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("レベル:B+")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("状態述語")] = 1
    base_phrase_feature_labels[1, 3, BASE_PHRASE_FEATURES.index("節-機能-原因・理由")] = 1
    base_phrase_feature_labels[1, 4, :] = IGNORE_INDEX  # ので
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1  # 頼み
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("時制:過去")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1
    base_phrase_feature_labels[1, 5, BASE_PHRASE_FEATURES.index("敬語:丁寧表現")] = 1
    base_phrase_feature_labels[1, 6:, :] = IGNORE_INDEX  # ました 〜
    assert base_phrase_feature_labels[0].tolist() == dataset[0].base_phrase_feature_labels
    assert base_phrase_feature_labels[1].tolist() == dataset[1].base_phrase_feature_labels

    dependency_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    dependency_labels[0, 0] = 2  # 太郎 -> 次郎
    dependency_labels[0, 1] = 0  # と -> 太郎
    dependency_labels[0, 2] = 5  # 次郎 -> けんか
    dependency_labels[0, 3] = 2  # は -> 次郎
    dependency_labels[0, 4] = 5  # よく -> けんか
    dependency_labels[0, 5] = dataset.special_token2index["[ROOT]"]  # けんか -> [ROOT]
    dependency_labels[0, 6] = 5  # する -> けんか
    dependency_labels[1, 0] = 1  # 辛い -> ラーメン
    dependency_labels[1, 1] = 3  # ラーメン -> 好きな
    dependency_labels[1, 2] = 1  # が -> ラーメン
    dependency_labels[1, 3] = 5  # 好きな -> 頼み
    dependency_labels[1, 4] = 3  # ので -> 好きな
    dependency_labels[1, 5] = dataset.special_token2index["[ROOT]"]  # 頼み -> [ROOT]
    dependency_labels[1, 6] = 5  # ました -> 頼み
    assert dependency_labels[0].tolist() == dataset[0].dependency_labels
    assert dependency_labels[1].tolist() == dataset[1].dependency_labels

    dependency_type_labels = torch.full((num_examples, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    dependency_type_labels[0, 0] = DEPENDENCY_TYPES.index(DepType.PARALLEL)  # 太郎 -> 次郎
    dependency_type_labels[0, 1] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # が -> 太郎
    dependency_type_labels[0, 2] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 次郎 -> けんか
    dependency_type_labels[0, 3] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # は -> 次郎
    dependency_type_labels[0, 4] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # よく -> けんか
    dependency_type_labels[0, 5] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # けんか -> [ROOT]
    dependency_type_labels[0, 6] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # する -> けんか
    dependency_type_labels[1, 0] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 辛い -> ラーメン
    dependency_type_labels[1, 1] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ラーメン -> 好きな
    dependency_type_labels[1, 2] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # が -> ラーメン
    dependency_type_labels[1, 3] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 好きな -> 頼み
    dependency_type_labels[1, 4] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ので -> 好きな
    dependency_type_labels[1, 5] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # 頼み -> [ROOT]
    dependency_type_labels[1, 6] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)  # ました -> 頼み
    assert dependency_type_labels[0].tolist() == dataset[0].dependency_type_labels
    assert dependency_type_labels[1].tolist() == dataset[1].dependency_type_labels

    flatten_rels = [r for cohesion_utils in dataset.cohesion_task2utils.values() for r in cohesion_utils.rels]
    cohesion_labels = torch.zeros((num_examples, len(flatten_rels), max_seq_length, max_seq_length), dtype=torch.long)
    cohesion_labels[0, flatten_rels.index("ノ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ノ 太郎
    cohesion_labels[0, flatten_rels.index("="), 0, dataset.special_token2index["[NA]"]] = 1  # 太郎
    cohesion_labels[0, flatten_rels.index("ノ"), 2, dataset.special_token2index["[NULL]"]] = 1  # φ ノ 次郎
    cohesion_labels[0, flatten_rels.index("="), 2, dataset.special_token2index["[NA]"]] = 1  # 次郎
    cohesion_labels[0, flatten_rels.index("ガ"), 5, 2] = 1  # 次郎 ガ けんか
    cohesion_labels[0, flatten_rels.index("ヲ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ けんか
    cohesion_labels[0, flatten_rels.index("ニ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ニ けんか
    cohesion_labels[0, flatten_rels.index("ガ２"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ けんか
    cohesion_labels[1, flatten_rels.index("ガ"), 0, 1] = 1  # ラーメン ガ 辛い
    cohesion_labels[1, flatten_rels.index("ヲ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ 辛い
    cohesion_labels[1, flatten_rels.index("ニ"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 辛い
    cohesion_labels[1, flatten_rels.index("ガ２"), 0, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ 辛い
    cohesion_labels[1, flatten_rels.index("ノ"), 1, dataset.special_token2index["[NULL]"]] = 1  # φ ノ ラーメン
    cohesion_labels[1, flatten_rels.index("="), 1, dataset.special_token2index["[NA]"]] = 1  # ラーメン
    cohesion_labels[1, flatten_rels.index("ガ"), 3, 1] = 1  # ラーメン ガ 好き
    cohesion_labels[1, flatten_rels.index("ヲ"), 3, dataset.special_token2index["[NULL]"]] = 1  # φ ヲ 好き
    cohesion_labels[1, flatten_rels.index("ニ"), 3, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 好き
    cohesion_labels[1, flatten_rels.index("ガ２"), 3, dataset.special_token2index["著者"]] = 1  # 著者 ガ２ 好き
    cohesion_labels[1, flatten_rels.index("ガ"), 5, dataset.special_token2index["著者"]] = 1  # 著者 ガ 頼み
    cohesion_labels[1, flatten_rels.index("ヲ"), 5, 1] = 1  # ラーメン ヲ 頼み
    cohesion_labels[1, flatten_rels.index("ニ"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ニ 頼み
    cohesion_labels[1, flatten_rels.index("ガ２"), 5, dataset.special_token2index["[NULL]"]] = 1  # φ ガ２ 頼み
    assert cohesion_labels[0].tolist() == dataset[0].cohesion_labels
    assert cohesion_labels[1].tolist() == dataset[1].cohesion_labels

    discourse_labels = torch.full((num_examples, max_seq_length, max_seq_length), IGNORE_INDEX, dtype=torch.long)
    discourse_labels[1, 3, 3] = DISCOURSE_RELATIONS.index("談話関係なし")  # 好きな - 好きな|談話関係なし
    discourse_labels[1, 3, 5] = DISCOURSE_RELATIONS.index("原因・理由")  # 好きな - 頼み|原因・理由
    discourse_labels[1, 5, 3] = DISCOURSE_RELATIONS.index("談話関係なし")  # 頼み - 好きな|談話関係なし
    discourse_labels[1, 5, 5] = DISCOURSE_RELATIONS.index("談話関係なし")  # 頼み - 頼み|談話関係なし
    assert discourse_labels[0].tolist() == dataset[0].discourse_labels
    assert discourse_labels[1].tolist() == dataset[1].discourse_labels
