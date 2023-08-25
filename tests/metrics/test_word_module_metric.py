from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from rhoknp import Document
from rhoknp.props import DepType
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordDataset
from kwja.metrics import WordModuleMetric
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    NE_TAGS,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
    WordTask,
)


def test_word_module_metric(
    data_dir: Path,
    word_tokenizer: PreTrainedTokenizerBase,
    dataset_kwargs: Dict[str, Any],
) -> None:
    path = data_dir / "datasets" / "word_files"
    max_seq_length = 20
    dataset = WordDataset(str(path), word_tokenizer, max_seq_length, document_split_stride=1, **dataset_kwargs)
    dataset.examples[1].load_discourse_document(Document.from_knp(path.joinpath("1.knp").read_text()))
    reading_id2reading = {v: k for k, v in dataset.reading2reading_id.items()}
    training_tasks = [
        WordTask.READING_PREDICTION,
        WordTask.MORPHOLOGICAL_ANALYSIS,
        WordTask.WORD_FEATURE_TAGGING,
        WordTask.NER,
        WordTask.BASE_PHRASE_FEATURE_TAGGING,
        WordTask.DEPENDENCY_PARSING,
        WordTask.COHESION_ANALYSIS,
        WordTask.DISCOURSE_PARSING,
    ]

    metric = WordModuleMetric(max_seq_length)
    metric.set_properties(
        {
            "dataset": dataset,
            "reading_id2reading": reading_id2reading,
            "training_tasks": training_tasks,
        }
    )
    metric.update(
        {
            "example_ids": torch.empty(0),  # dummy
            "reading_predictions": torch.empty(0),
            "reading_subword_map": torch.empty(0),
            "pos_logits": torch.empty(0),
            "subpos_logits": torch.empty(0),
            "conjtype_logits": torch.empty(0),
            "conjform_logits": torch.empty(0),
            "word_feature_probabilities": torch.empty(0),
            "ne_predictions": torch.empty(0),
            "base_phrase_feature_probabilities": torch.empty(0),
            "dependency_predictions": torch.empty(0),
            "dependency_type_predictions": torch.empty(0),
            "cohesion_logits": torch.empty(0),
            "discourse_predictions": torch.empty(0),
            "discourse_labels": torch.empty(0),
        }
    )

    num_examples = len(dataset)
    metric.example_ids = torch.arange(num_examples, dtype=torch.long)

    metric.reading_predictions = torch.zeros((num_examples, max_seq_length), dtype=torch.long)
    # [:, 0] = [CLS]
    metric.reading_predictions[0, 1] = dataset.reading2reading_id["たろう"]  # 太郎 -> たろう
    metric.reading_predictions[0, 2] = dataset.reading2reading_id["[ID]"]  # と
    metric.reading_predictions[0, 3] = dataset.reading2reading_id["じろう"]  # 次郎 -> じろう
    metric.reading_predictions[0, 4] = dataset.reading2reading_id["[ID]"]  # は
    metric.reading_predictions[0, 5] = dataset.reading2reading_id["[ID]"]  # よく
    metric.reading_predictions[0, 6] = dataset.reading2reading_id["[ID]"]  # けん
    metric.reading_predictions[0, 7] = dataset.reading2reading_id["[ID]"]  # か
    metric.reading_predictions[0, 8] = dataset.reading2reading_id["[ID]"]  # する
    metric.reading_predictions[1, 1] = dataset.reading2reading_id["からい"]  # 辛い -> からい
    metric.reading_predictions[1, 2] = dataset.reading2reading_id["らーめん"]  # ラーメン -> らーめん
    metric.reading_predictions[1, 3] = dataset.reading2reading_id["[ID]"]  # が
    metric.reading_predictions[1, 4] = dataset.reading2reading_id["すきな"]  # 好きな -> すきな
    metric.reading_predictions[1, 5] = dataset.reading2reading_id["[ID]"]  # ので
    metric.reading_predictions[1, 6] = dataset.reading2reading_id["たのみ"]  # 頼み -> たのみ
    metric.reading_predictions[1, 7] = dataset.reading2reading_id["[ID]"]  # ました

    metric.reading_subword_map = torch.zeros((num_examples, max_seq_length, max_seq_length), dtype=torch.bool)
    metric.reading_subword_map[0, 0, 1] = True
    metric.reading_subword_map[0, 1, 2] = True
    metric.reading_subword_map[0, 2, 3] = True
    metric.reading_subword_map[0, 3, 4] = True
    metric.reading_subword_map[0, 4, 5] = True
    metric.reading_subword_map[0, 5, 6] = True  # けんか -> けん
    metric.reading_subword_map[0, 5, 7] = True  # けんか -> か
    metric.reading_subword_map[0, 6, 8] = True
    metric.reading_subword_map[1, 0, 1] = True
    metric.reading_subword_map[1, 1, 2] = True
    metric.reading_subword_map[1, 2, 3] = True
    metric.reading_subword_map[1, 3, 4] = True
    metric.reading_subword_map[1, 4, 5] = True
    metric.reading_subword_map[1, 5, 6] = True
    metric.reading_subword_map[1, 6, 7] = True

    metric.pos_logits = torch.zeros((num_examples, max_seq_length, len(POS_TAGS)), dtype=torch.float)
    metric.pos_logits[0, 0, POS_TAGS.index("名詞")] = 1.0  # 太郎
    metric.pos_logits[0, 1, POS_TAGS.index("助詞")] = 1.0  # と
    metric.pos_logits[0, 2, POS_TAGS.index("名詞")] = 1.0  # 次郎
    metric.pos_logits[0, 3, POS_TAGS.index("助詞")] = 1.0  # は
    metric.pos_logits[0, 4, POS_TAGS.index("副詞")] = 1.0  # よく
    metric.pos_logits[0, 5, POS_TAGS.index("名詞")] = 1.0  # けんか
    metric.pos_logits[0, 6, POS_TAGS.index("動詞")] = 1.0  # する
    metric.pos_logits[1, 0, POS_TAGS.index("形容詞")] = 1.0  # 辛い
    metric.pos_logits[1, 1, POS_TAGS.index("名詞")] = 1.0  # ラーメン
    metric.pos_logits[1, 2, POS_TAGS.index("助詞")] = 1.0  # が
    metric.pos_logits[1, 3, POS_TAGS.index("形容詞")] = 1.0  # 好きな
    metric.pos_logits[1, 4, POS_TAGS.index("助動詞")] = 1.0  # ので
    metric.pos_logits[1, 5, POS_TAGS.index("動詞")] = 1.0  # 頼み
    metric.pos_logits[1, 6, POS_TAGS.index("接尾辞")] = 1.0  # ました

    metric.subpos_logits = torch.zeros((num_examples, max_seq_length, len(SUBPOS_TAGS)), dtype=torch.float)
    metric.subpos_logits[0, 0, SUBPOS_TAGS.index("人名")] = 1.0  # 太郎
    metric.subpos_logits[0, 1, SUBPOS_TAGS.index("格助詞")] = 1.0  # と
    metric.subpos_logits[0, 2, SUBPOS_TAGS.index("人名")] = 1.0  # 次郎
    metric.subpos_logits[0, 3, SUBPOS_TAGS.index("副助詞")] = 1.0  # は
    metric.subpos_logits[0, 4, SUBPOS_TAGS.index("*")] = 1.0  # よく
    metric.subpos_logits[0, 5, SUBPOS_TAGS.index("サ変名詞")] = 1.0  # けんか
    metric.subpos_logits[0, 6, SUBPOS_TAGS.index("*")] = 1.0  # する
    metric.subpos_logits[1, 0, SUBPOS_TAGS.index("*")] = 1.0  # 辛い
    metric.subpos_logits[1, 1, SUBPOS_TAGS.index("普通名詞")] = 1.0  # ラーメン
    metric.subpos_logits[1, 2, SUBPOS_TAGS.index("格助詞")] = 1.0  # が
    metric.subpos_logits[1, 3, SUBPOS_TAGS.index("*")] = 1.0  # 好きな
    metric.subpos_logits[1, 4, SUBPOS_TAGS.index("*")] = 1.0  # ので
    metric.subpos_logits[1, 5, SUBPOS_TAGS.index("*")] = 1.0  # 頼み
    metric.subpos_logits[1, 6, SUBPOS_TAGS.index("動詞性接尾辞")] = 1.0  # ました

    metric.conjtype_logits = torch.zeros((num_examples, max_seq_length, len(CONJTYPE_TAGS)), dtype=torch.float)
    metric.conjtype_logits[0, 0, CONJTYPE_TAGS.index("*")] = 1.0  # 太郎
    metric.conjtype_logits[0, 1, CONJTYPE_TAGS.index("*")] = 1.0  # と
    metric.conjtype_logits[0, 2, CONJTYPE_TAGS.index("*")] = 1.0  # 次郎
    metric.conjtype_logits[0, 3, CONJTYPE_TAGS.index("*")] = 1.0  # は
    metric.conjtype_logits[0, 4, CONJTYPE_TAGS.index("*")] = 1.0  # よく
    metric.conjtype_logits[0, 5, CONJTYPE_TAGS.index("*")] = 1.0  # けんか
    metric.conjtype_logits[0, 6, CONJTYPE_TAGS.index("サ変動詞")] = 1.0  # する
    metric.conjtype_logits[1, 0, CONJTYPE_TAGS.index("イ形容詞アウオ段")] = 1.0  # 辛い
    metric.conjtype_logits[1, 1, CONJTYPE_TAGS.index("*")] = 1.0  # ラーメン
    metric.conjtype_logits[1, 2, CONJTYPE_TAGS.index("*")] = 1.0  # が
    metric.conjtype_logits[1, 3, CONJTYPE_TAGS.index("ナ形容詞")] = 1.0  # 好きな
    metric.conjtype_logits[1, 4, CONJTYPE_TAGS.index("ナ形容詞")] = 1.0  # ので
    metric.conjtype_logits[1, 5, CONJTYPE_TAGS.index("子音動詞マ行")] = 1.0  # 頼み
    metric.conjtype_logits[1, 6, CONJTYPE_TAGS.index("動詞性接尾辞ます型")] = 1.0  # ました

    metric.conjform_logits = torch.zeros((num_examples, max_seq_length, len(CONJFORM_TAGS)), dtype=torch.float)
    metric.conjform_logits[0, 0, CONJFORM_TAGS.index("*")] = 1.0  # 太郎
    metric.conjform_logits[0, 1, CONJFORM_TAGS.index("*")] = 1.0  # と
    metric.conjform_logits[0, 2, CONJFORM_TAGS.index("*")] = 1.0  # 次郎
    metric.conjform_logits[0, 3, CONJFORM_TAGS.index("*")] = 1.0  # は
    metric.conjform_logits[0, 4, CONJFORM_TAGS.index("*")] = 1.0  # よく
    metric.conjform_logits[0, 5, CONJFORM_TAGS.index("*")] = 1.0  # けんか
    metric.conjform_logits[0, 6, CONJFORM_TAGS.index("基本形")] = 1.0  # する
    metric.conjform_logits[1, 0, CONJFORM_TAGS.index("基本形")] = 1.0  # 辛い
    metric.conjform_logits[1, 1, CONJFORM_TAGS.index("*")] = 1.0  # ラーメン
    metric.conjform_logits[1, 2, CONJFORM_TAGS.index("*")] = 1.0  # が
    metric.conjform_logits[1, 3, CONJFORM_TAGS.index("ダ列基本連体形")] = 1.0  # 好きな
    metric.conjform_logits[1, 4, CONJFORM_TAGS.index("ダ列タ系連用テ形")] = 1.0  # ので
    metric.conjform_logits[1, 5, CONJFORM_TAGS.index("基本連用形")] = 1.0  # 頼み
    metric.conjform_logits[1, 6, CONJFORM_TAGS.index("タ形")] = 1.0  # ました

    metric.word_feature_probabilities = torch.zeros(
        (num_examples, max_seq_length, len(WORD_FEATURES)), dtype=torch.float
    )
    metric.word_feature_probabilities[0, 0, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 太郎
    metric.word_feature_probabilities[0, 1, WORD_FEATURES.index("基本句-区切")] = 1.0  # と
    metric.word_feature_probabilities[0, 1, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[0, 2, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 次郎
    metric.word_feature_probabilities[0, 3, WORD_FEATURES.index("基本句-区切")] = 1.0  # は
    metric.word_feature_probabilities[0, 3, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[0, 4, WORD_FEATURES.index("基本句-主辞")] = 1.0  # よく
    metric.word_feature_probabilities[0, 4, WORD_FEATURES.index("基本句-区切")] = 1.0
    metric.word_feature_probabilities[0, 4, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[0, 4, WORD_FEATURES.index("用言表記先頭")] = 1.0
    metric.word_feature_probabilities[0, 4, WORD_FEATURES.index("用言表記末尾")] = 1.0
    metric.word_feature_probabilities[0, 5, WORD_FEATURES.index("基本句-主辞")] = 1.0  # けんか
    metric.word_feature_probabilities[0, 5, WORD_FEATURES.index("用言表記先頭")] = 1.0
    metric.word_feature_probabilities[0, 5, WORD_FEATURES.index("用言表記末尾")] = 1.0
    metric.word_feature_probabilities[0, 6, WORD_FEATURES.index("基本句-区切")] = 1.0  # する
    metric.word_feature_probabilities[0, 6, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[1, 0, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 辛い
    metric.word_feature_probabilities[1, 0, WORD_FEATURES.index("基本句-区切")] = 1.0
    metric.word_feature_probabilities[1, 0, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[1, 0, WORD_FEATURES.index("用言表記先頭")] = 1.0
    metric.word_feature_probabilities[1, 0, WORD_FEATURES.index("用言表記末尾")] = 1.0
    metric.word_feature_probabilities[1, 1, WORD_FEATURES.index("基本句-主辞")] = 1.0  # ラーメン
    metric.word_feature_probabilities[1, 2, WORD_FEATURES.index("基本句-区切")] = 1.0  # が
    metric.word_feature_probabilities[1, 2, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[1, 3, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 好きな
    metric.word_feature_probabilities[1, 3, WORD_FEATURES.index("用言表記先頭")] = 1.0
    metric.word_feature_probabilities[1, 3, WORD_FEATURES.index("用言表記末尾")] = 1.0
    metric.word_feature_probabilities[1, 4, WORD_FEATURES.index("基本句-区切")] = 1.0  # ので
    metric.word_feature_probabilities[1, 4, WORD_FEATURES.index("文節-区切")] = 1.0
    metric.word_feature_probabilities[1, 5, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 頼み
    metric.word_feature_probabilities[1, 5, WORD_FEATURES.index("用言表記先頭")] = 1.0
    metric.word_feature_probabilities[1, 5, WORD_FEATURES.index("用言表記末尾")] = 1.0
    metric.word_feature_probabilities[1, 6, WORD_FEATURES.index("基本句-区切")] = 1.0  # ました
    metric.word_feature_probabilities[1, 6, WORD_FEATURES.index("文節-区切")] = 1.0

    metric.ne_predictions = torch.zeros((num_examples, max_seq_length), dtype=torch.long)
    metric.ne_predictions[0, 0] = NE_TAGS.index("B-PERSON")  # 太郎
    metric.ne_predictions[0, 1] = NE_TAGS.index("O")  # と
    metric.ne_predictions[0, 2] = NE_TAGS.index("B-PERSON")  # 次郎
    metric.ne_predictions[0, 3:7] = NE_TAGS.index("O")  # は よく けんか する
    metric.ne_predictions[1, :7] = NE_TAGS.index("O")  # 辛い ラーメン が 好きな ので 頼み ました

    metric.base_phrase_feature_probabilities = torch.zeros(
        (num_examples, max_seq_length, len(BASE_PHRASE_FEATURES)), dtype=torch.float
    )
    metric.base_phrase_feature_probabilities[0, 0, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # 太郎
    metric.base_phrase_feature_probabilities[0, 0, BASE_PHRASE_FEATURES.index("SM-主体")] = 1.0
    metric.base_phrase_feature_probabilities[0, 2, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # 次郎
    metric.base_phrase_feature_probabilities[0, 2, BASE_PHRASE_FEATURES.index("SM-主体")] = 1.0
    metric.base_phrase_feature_probabilities[0, 4, BASE_PHRASE_FEATURES.index("修飾")] = 1.0  # よく
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1.0  # けんか
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    metric.base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1.0
    metric.base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("用言:形")] = 1.0  # 辛い
    metric.base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    metric.base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("レベル:B-")] = 1.0
    metric.base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("状態述語")] = 1.0
    metric.base_phrase_feature_probabilities[1, 1, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # ラーメン
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("用言:形")] = 1.0  # 好きな
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("レベル:B+")] = 1.0
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("状態述語")] = 1.0
    metric.base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-機能-原因・理由")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1.0  # 頼み
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("時制:過去")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1.0
    metric.base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("敬語:丁寧表現")] = 1.0

    dependency_topk = 2
    metric.dependency_predictions = torch.zeros((num_examples, max_seq_length, dependency_topk), dtype=torch.long)
    metric.dependency_predictions[0, 0] = torch.as_tensor([2, 5])  # 太郎 -> 次郎, けんか
    metric.dependency_predictions[0, 1] = torch.as_tensor([0, 5])  # と -> 太郎, けんか
    metric.dependency_predictions[0, 2] = torch.as_tensor([5, 6])  # 次郎 -> けんか, する
    metric.dependency_predictions[0, 3] = torch.as_tensor([2, 5])  # は -> 次郎, けんか
    metric.dependency_predictions[0, 4] = torch.as_tensor([5, 6])  # よく -> けんか, する
    metric.dependency_predictions[0, 5] = torch.as_tensor(
        [dataset.examples[0].special_token_indexer.get_morpheme_level_index("[ROOT]"), 6]
    )  # けんか -> [ROOT], する
    metric.dependency_predictions[0, 6] = torch.as_tensor([5, 4])  # する -> けんか, よく
    metric.dependency_predictions[1, 0] = torch.as_tensor([1, 3])  # 辛い -> ラーメン, 好きな
    metric.dependency_predictions[1, 1] = torch.as_tensor([3, 5])  # ラーメン -> 好きな, 頼み
    metric.dependency_predictions[1, 2] = torch.as_tensor([1, 3])  # が -> ラーメン, 好きな
    metric.dependency_predictions[1, 3] = torch.as_tensor([5, 6])  # 好きな -> 頼み, ました
    metric.dependency_predictions[1, 4] = torch.as_tensor([3, 5])  # ので -> 好きな, 頼み
    metric.dependency_predictions[1, 5] = torch.as_tensor(
        [dataset.examples[1].special_token_indexer.get_morpheme_level_index("[ROOT]"), 6]
    )  # 頼み -> [ROOT], ました
    metric.dependency_predictions[1, 6] = torch.as_tensor([5, 3])  # ました -> 頼み, 好きな

    metric.dependency_type_predictions = torch.zeros((num_examples, max_seq_length, dependency_topk), dtype=torch.long)
    d = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    p = DEPENDENCY_TYPES.index(DepType.PARALLEL)
    metric.dependency_type_predictions[0, 0] = torch.as_tensor([p, d])  # 太郎 -> 次郎, けんか
    metric.dependency_type_predictions[0, 1] = torch.as_tensor([d, d])  # が -> 太郎, けんか
    metric.dependency_type_predictions[0, 2] = torch.as_tensor([d, d])  # 次郎 -> けんか, する
    metric.dependency_type_predictions[0, 3] = torch.as_tensor([d, d])  # は -> 次郎, けんか
    metric.dependency_type_predictions[0, 4] = torch.as_tensor([d, d])  # よく -> けんか, する
    metric.dependency_type_predictions[0, 5] = torch.as_tensor([d, d])  # けんか -> [ROOT], する
    metric.dependency_type_predictions[0, 6] = torch.as_tensor([d, d])  # する -> けんか, よく
    metric.dependency_type_predictions[1, 0] = torch.as_tensor([d, d])  # 辛い -> ラーメン, 好きな
    metric.dependency_type_predictions[1, 1] = torch.as_tensor([d, d])  # ラーメン -> 好きな, 頼み
    metric.dependency_type_predictions[1, 2] = torch.as_tensor([d, d])  # が -> ラーメン, 好きな
    metric.dependency_type_predictions[1, 3] = torch.as_tensor([d, d])  # 好きな -> 頼み, ました
    metric.dependency_type_predictions[1, 4] = torch.as_tensor([d, d])  # ので -> 好きな, 頼み
    metric.dependency_type_predictions[1, 5] = torch.as_tensor([d, d])  # 頼み -> [ROOT], ました
    metric.dependency_type_predictions[1, 6] = torch.as_tensor([d, d])  # ました -> 頼み, 好きな

    flatten_rels = [r for rels in dataset.cohesion_task2rels.values() for r in rels]
    metric.cohesion_logits = torch.zeros(
        (num_examples, len(flatten_rels), max_seq_length, max_seq_length), dtype=torch.float
    )
    for i in range(num_examples):
        for j, rel in enumerate(flatten_rels):
            if rel == "=":
                k = dataset.examples[i].special_token_indexer.get_morpheme_level_index("[NA]")
            else:
                k = dataset.examples[i].special_token_indexer.get_morpheme_level_index("[NULL]")
            metric.cohesion_logits[i, j, :, k] = 1.0
    metric.cohesion_logits[0, flatten_rels.index("ガ"), 5, 2] = 2.0  # 次郎 ガ けんか
    metric.cohesion_logits[1, flatten_rels.index("ガ"), 0, 1] = 2.0  # ラーメン ガ 辛い
    metric.cohesion_logits[1, flatten_rels.index("ガ"), 3, 1] = 2.0  # ラーメン ガ 好き
    metric.cohesion_logits[
        1, flatten_rels.index("ガ２"), 3, dataset.examples[1].special_token_indexer.get_morpheme_level_index("[著者]")
    ] = 2.0  # 著者 ガ２ 好き
    metric.cohesion_logits[
        1, flatten_rels.index("ガ"), 5, dataset.examples[1].special_token_indexer.get_morpheme_level_index("[著者]")
    ] = 2.0  # 著者 ガ 頼み
    metric.cohesion_logits[1, flatten_rels.index("ヲ"), 5, 1] = 2.0  # ラーメン ヲ 頼み

    metric.discourse_predictions = torch.zeros((num_examples, max_seq_length, max_seq_length), dtype=torch.long)
    metric.discourse_predictions[1, 3, 5] = DISCOURSE_RELATIONS.index("原因・理由")  # 好きな - 頼み|原因・理由
    metric.discourse_labels = torch.stack(
        [torch.as_tensor(dataset[eid].discourse_labels) for eid in metric.example_ids], dim=0
    )

    metrics = metric.compute()

    # num_corrects = , total =
    assert metrics["reading_prediction_accuracy"] == 1.0

    # tp = , fp = , fn =
    assert metrics["pos_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["subpos_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["conjtype_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["conjform_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["morphological_analysis_f1"] == pytest.approx(1.0)

    assert metrics["macro_word_feature_tagging_f1"] == pytest.approx(1.0)
    assert metrics["micro_word_feature_tagging_f1"] == pytest.approx(1.0)

    # tp = , fp = , fn =
    assert metrics["ner_f1"] == pytest.approx(1.0)

    assert metrics["macro_base_phrase_feature_tagging_f1"] == pytest.approx(1.0)
    assert metrics["micro_base_phrase_feature_tagging_f1"] == pytest.approx(1.0)

    # tp = , fp = , fn =
    assert metrics["base_phrase_LAS_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["base_phrase_UAS_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["morpheme_LAS_f1"] == pytest.approx(1.0)
    # tp = , fp = , fn =
    assert metrics["morpheme_UAS_f1"] == pytest.approx(1.0)

    assert metrics["pas_f1"] == pytest.approx(1.0)
    assert metrics["bridging_f1"] == pytest.approx(0.0)
    assert metrics["coreference_f1"] == pytest.approx(0.0)
    assert metrics["cohesion_analysis_f1"] == pytest.approx(1 / 3)

    assert metrics["discourse_parsing_f1"] == pytest.approx(1.0)
