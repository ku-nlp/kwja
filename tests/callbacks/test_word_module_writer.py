import tempfile
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, Union

import lightning as L
import pytest
import torch
from rhoknp.props import DepType
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.callbacks import WordModuleWriter
from kwja.datamodule.datasets import WordInferenceDataset
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
from kwja.utils.jumandic import JumanDic
from kwja.utils.reading_prediction import get_reading2reading_id

AMBIG_SURF_SPECS = [
    {
        "conjtype": "イ形容詞アウオ段",
        "conjform": "エ基本形",
    },
    {
        "conjtype": "イ形容詞イ段",
        "conjform": "エ基本形",
    },
    {
        "conjtype": "イ形容詞イ段特殊",
        "conjform": "エ基本形",
    },
]


class MockTrainer:
    def __init__(self, predict_dataloaders: list[DataLoader]) -> None:
        self.predict_dataloaders = predict_dataloaders


def build_dummy_jumandic() -> JumanDic:
    with tempfile.TemporaryDirectory() as jumandic_dir:
        JumanDic.build(
            Path(jumandic_dir),
            [
                ["太郎", "たろう", "太郎", "名詞", "人名", "*", "*", "太郎/たろう", "代表表記:太郎/たろう 人名"],
                ["次郎", "じろう", "次郎", "名詞", "人名", "*", "*", "次郎/じろう", "代表表記:次郎/じろう 人名"],
                ["けんか", "けんか", "けんか", "名詞", "サ変名詞", "*", "*", "喧嘩/けんか", "代表表記:喧嘩/けんか"],
                ["けんか", "けんか", "けんか", "名詞", "サ変名詞", "*", "*", "献花/けんか", "代表表記:献花/けんか"],
            ],
        )
        return JumanDic(Path(jumandic_dir))


@pytest.mark.parametrize(
    "destination",
    [
        None,
        Path(tempfile.TemporaryDirectory().name) / Path("word_prediction.knp"),
        str(Path(tempfile.TemporaryDirectory().name) / Path("word_prediction.knp")),
    ],
)
def test_init(destination: Optional[Union[str, Path]]) -> None:
    _ = WordModuleWriter(AMBIG_SURF_SPECS, destination=destination)


def test_write_on_batch_end(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    doc_id_prefix = "test"
    juman_text = dedent(
        f"""\
        # S-ID:{doc_id_prefix}-0-0 kwja:{version("kwja")}
        太郎 _ 太郎 未定義語 15 その他 1 * 0 * 0
        と _ と 未定義語 15 その他 1 * 0 * 0
        次郎 _ 次郎 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        よく _ よく 未定義語 15 その他 1 * 0 * 0
        けんか _ けんか 未定義語 15 その他 1 * 0 * 0
        する _ する 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:{doc_id_prefix}-1-0 kwja:{version("kwja")}
        辛い _ 辛い 未定義語 15 その他 1 * 0 * 0
        ラーメン _ ラーメン 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        好きな _ 好きな 未定義語 15 その他 1 * 0 * 0
        ので _ ので 未定義語 15 その他 1 * 0 * 0
        頼み _ 頼み 未定義語 15 その他 1 * 0 * 0
        ました _ ました 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    max_seq_length = 32  # >= 17
    dataset = WordInferenceDataset(
        word_tokenizer, max_seq_length, document_split_stride=1, juman_file=Path(juman_file.name), **dataset_kwargs
    )
    num_examples = len(dataset)

    trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])

    module = L.LightningModule()
    module.training_tasks = [
        WordTask.READING_PREDICTION,
        WordTask.MORPHOLOGICAL_ANALYSIS,
        WordTask.WORD_FEATURE_TAGGING,
        WordTask.NER,
        WordTask.BASE_PHRASE_FEATURE_TAGGING,
        WordTask.DEPENDENCY_PARSING,
        WordTask.COHESION_ANALYSIS,
        WordTask.DISCOURSE_RELATION_ANALYSIS,
    ]

    reading2reading_id = get_reading2reading_id()
    reading_logits = torch.zeros((num_examples, max_seq_length, len(reading2reading_id)), dtype=torch.float)
    reading_logits[0, 1, reading2reading_id["たろう"]] = 1.0  # 太郎 -> たろう
    reading_logits[0, 2, reading2reading_id["[ID]"]] = 1.0  # と
    reading_logits[0, 3, reading2reading_id["じろう"]] = 1.0  # 次郎 -> じろう
    reading_logits[0, 4, reading2reading_id["[ID]"]] = 1.0  # は
    reading_logits[0, 5, reading2reading_id["[ID]"]] = 1.0  # よく
    reading_logits[0, 6, reading2reading_id["[ID]"]] = 1.0  # けん
    reading_logits[0, 7, reading2reading_id["[ID]"]] = 1.0  # か
    reading_logits[0, 8, reading2reading_id["[ID]"]] = 1.0  # する
    reading_logits[1, 1, reading2reading_id["からい"]] = 1.0  # 辛い -> からい
    reading_logits[1, 2, reading2reading_id["らーめん"]] = 1.0  # ラーメン -> らーめん
    reading_logits[1, 3, reading2reading_id["[ID]"]] = 1.0  # が
    reading_logits[1, 4, reading2reading_id["すきな"]] = 1.0  # 好きな -> すきな
    reading_logits[1, 5, reading2reading_id["[ID]"]] = 1.0  # ので
    reading_logits[1, 6, reading2reading_id["たのみ"]] = 1.0  # 頼み -> たのみ
    reading_logits[1, 7, reading2reading_id["[ID]"]] = 1.0  # ました

    # (b, word, token)
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

    pos_logits = torch.zeros((num_examples, max_seq_length, len(POS_TAGS)), dtype=torch.float)
    pos_logits[0, 0, POS_TAGS.index("名詞")] = 1.0  # 太郎
    pos_logits[0, 1, POS_TAGS.index("助詞")] = 1.0  # と
    pos_logits[0, 2, POS_TAGS.index("名詞")] = 1.0  # 次郎
    pos_logits[0, 3, POS_TAGS.index("助詞")] = 1.0  # は
    pos_logits[0, 4, POS_TAGS.index("副詞")] = 1.0  # よく
    pos_logits[0, 5, POS_TAGS.index("名詞")] = 1.0  # けんか
    pos_logits[0, 6, POS_TAGS.index("動詞")] = 1.0  # する
    pos_logits[1, 0, POS_TAGS.index("形容詞")] = 1.0  # 辛い
    pos_logits[1, 1, POS_TAGS.index("名詞")] = 1.0  # ラーメン
    pos_logits[1, 2, POS_TAGS.index("助詞")] = 1.0  # が
    pos_logits[1, 3, POS_TAGS.index("形容詞")] = 1.0  # 好きな
    pos_logits[1, 4, POS_TAGS.index("助動詞")] = 1.0  # ので
    pos_logits[1, 5, POS_TAGS.index("動詞")] = 1.0  # 頼み
    pos_logits[1, 6, POS_TAGS.index("接尾辞")] = 1.0  # ました

    subpos_logits = torch.zeros((num_examples, max_seq_length, len(SUBPOS_TAGS)), dtype=torch.float)
    subpos_logits[0, 0, SUBPOS_TAGS.index("人名")] = 1.0  # 太郎
    subpos_logits[0, 1, SUBPOS_TAGS.index("格助詞")] = 1.0  # と
    subpos_logits[0, 2, SUBPOS_TAGS.index("人名")] = 1.0  # 次郎
    subpos_logits[0, 3, SUBPOS_TAGS.index("副助詞")] = 1.0  # は
    subpos_logits[0, 4, SUBPOS_TAGS.index("*")] = 1.0  # よく
    subpos_logits[0, 5, SUBPOS_TAGS.index("サ変名詞")] = 1.0  # けんか
    subpos_logits[0, 6, SUBPOS_TAGS.index("*")] = 1.0  # する
    subpos_logits[1, 0, SUBPOS_TAGS.index("*")] = 1.0  # 辛い
    subpos_logits[1, 1, SUBPOS_TAGS.index("普通名詞")] = 1.0  # ラーメン
    subpos_logits[1, 2, SUBPOS_TAGS.index("格助詞")] = 1.0  # が
    subpos_logits[1, 3, SUBPOS_TAGS.index("*")] = 1.0  # 好きな
    subpos_logits[1, 4, SUBPOS_TAGS.index("*")] = 1.0  # ので
    subpos_logits[1, 5, SUBPOS_TAGS.index("*")] = 1.0  # 頼み
    subpos_logits[1, 6, SUBPOS_TAGS.index("動詞性接尾辞")] = 1.0  # ました

    conjtype_logits = torch.zeros((num_examples, max_seq_length, len(CONJTYPE_TAGS)), dtype=torch.float)
    conjtype_logits[0, 0, CONJTYPE_TAGS.index("*")] = 1.0  # 太郎
    conjtype_logits[0, 1, CONJTYPE_TAGS.index("*")] = 1.0  # と
    conjtype_logits[0, 2, CONJTYPE_TAGS.index("*")] = 1.0  # 次郎
    conjtype_logits[0, 3, CONJTYPE_TAGS.index("*")] = 1.0  # は
    conjtype_logits[0, 4, CONJTYPE_TAGS.index("*")] = 1.0  # よく
    conjtype_logits[0, 5, CONJTYPE_TAGS.index("*")] = 1.0  # けんか
    conjtype_logits[0, 6, CONJTYPE_TAGS.index("サ変動詞")] = 1.0  # する
    conjtype_logits[1, 0, CONJTYPE_TAGS.index("イ形容詞アウオ段")] = 1.0  # 辛い
    conjtype_logits[1, 1, CONJTYPE_TAGS.index("*")] = 1.0  # ラーメン
    conjtype_logits[1, 2, CONJTYPE_TAGS.index("*")] = 1.0  # が
    conjtype_logits[1, 3, CONJTYPE_TAGS.index("ナ形容詞")] = 1.0  # 好きな
    conjtype_logits[1, 4, CONJTYPE_TAGS.index("ナ形容詞")] = 1.0  # ので
    conjtype_logits[1, 5, CONJTYPE_TAGS.index("子音動詞マ行")] = 1.0  # 頼み
    conjtype_logits[1, 6, CONJTYPE_TAGS.index("動詞性接尾辞ます型")] = 1.0  # ました

    conjform_logits = torch.zeros((num_examples, max_seq_length, len(CONJFORM_TAGS)), dtype=torch.float)
    conjform_logits[0, 0, CONJFORM_TAGS.index("*")] = 1.0  # 太郎
    conjform_logits[0, 1, CONJFORM_TAGS.index("*")] = 1.0  # と
    conjform_logits[0, 2, CONJFORM_TAGS.index("*")] = 1.0  # 次郎
    conjform_logits[0, 3, CONJFORM_TAGS.index("*")] = 1.0  # は
    conjform_logits[0, 4, CONJFORM_TAGS.index("*")] = 1.0  # よく
    conjform_logits[0, 5, CONJFORM_TAGS.index("*")] = 1.0  # けんか
    conjform_logits[0, 6, CONJFORM_TAGS.index("基本形")] = 1.0  # する
    conjform_logits[1, 0, CONJFORM_TAGS.index("基本形")] = 1.0  # 辛い
    conjform_logits[1, 1, CONJFORM_TAGS.index("*")] = 1.0  # ラーメン
    conjform_logits[1, 2, CONJFORM_TAGS.index("*")] = 1.0  # が
    conjform_logits[1, 3, CONJFORM_TAGS.index("ダ列基本連体形")] = 1.0  # 好きな
    conjform_logits[1, 4, CONJFORM_TAGS.index("ダ列タ系連用テ形")] = 1.0  # ので
    conjform_logits[1, 5, CONJFORM_TAGS.index("基本連用形")] = 1.0  # 頼み
    conjform_logits[1, 6, CONJFORM_TAGS.index("タ形")] = 1.0  # ました

    word_feature_probabilities = torch.zeros((num_examples, max_seq_length, len(WORD_FEATURES)), dtype=torch.float)
    word_feature_probabilities[0, 0, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 太郎
    word_feature_probabilities[0, 1, WORD_FEATURES.index("基本句-区切")] = 1.0  # と
    word_feature_probabilities[0, 1, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[0, 2, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 次郎
    word_feature_probabilities[0, 3, WORD_FEATURES.index("基本句-区切")] = 1.0  # は
    word_feature_probabilities[0, 3, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[0, 4, WORD_FEATURES.index("基本句-主辞")] = 1.0  # よく
    word_feature_probabilities[0, 4, WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_probabilities[0, 4, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[0, 4, WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[0, 4, WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[0, 5, WORD_FEATURES.index("基本句-主辞")] = 1.0  # けんか
    word_feature_probabilities[0, 5, WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[0, 5, WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[0, 6, WORD_FEATURES.index("基本句-区切")] = 1.0  # する
    word_feature_probabilities[0, 6, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[1, 0, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 辛い
    word_feature_probabilities[1, 0, WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_probabilities[1, 0, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[1, 0, WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[1, 0, WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[1, 1, WORD_FEATURES.index("基本句-主辞")] = 1.0  # ラーメン
    word_feature_probabilities[1, 2, WORD_FEATURES.index("基本句-区切")] = 1.0  # が
    word_feature_probabilities[1, 2, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[1, 3, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 好きな
    word_feature_probabilities[1, 3, WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[1, 3, WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[1, 4, WORD_FEATURES.index("基本句-区切")] = 1.0  # ので
    word_feature_probabilities[1, 4, WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[1, 5, WORD_FEATURES.index("基本句-主辞")] = 1.0  # 頼み
    word_feature_probabilities[1, 5, WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[1, 5, WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[1, 6, WORD_FEATURES.index("基本句-区切")] = 1.0  # ました
    word_feature_probabilities[1, 6, WORD_FEATURES.index("文節-区切")] = 1.0

    ne_predictions = torch.full((num_examples, max_seq_length), NE_TAGS.index("O"), dtype=torch.long)
    ne_predictions[0, 0] = NE_TAGS.index("B-PERSON")  # 太郎
    ne_predictions[0, 2] = NE_TAGS.index("B-PERSON")  # 次郎

    base_phrase_feature_probabilities = torch.zeros(
        (num_examples, max_seq_length, len(BASE_PHRASE_FEATURES)), dtype=torch.float
    )
    base_phrase_feature_probabilities[0, 0, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # 太郎
    base_phrase_feature_probabilities[0, 0, BASE_PHRASE_FEATURES.index("SM-主体")] = 1.0
    base_phrase_feature_probabilities[0, 2, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # 次郎
    base_phrase_feature_probabilities[0, 2, BASE_PHRASE_FEATURES.index("SM-主体")] = 1.0
    base_phrase_feature_probabilities[0, 4, BASE_PHRASE_FEATURES.index("修飾")] = 1.0  # よく
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1.0  # けんか
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    base_phrase_feature_probabilities[0, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1.0
    base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("用言:形")] = 1.0  # 辛い
    base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("レベル:B-")] = 1.0
    base_phrase_feature_probabilities[1, 0, BASE_PHRASE_FEATURES.index("状態述語")] = 1.0
    base_phrase_feature_probabilities[1, 1, BASE_PHRASE_FEATURES.index("体言")] = 1.0  # ラーメン
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("用言:形")] = 1.0  # 好きな
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("レベル:B+")] = 1.0
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("状態述語")] = 1.0
    base_phrase_feature_probabilities[1, 3, BASE_PHRASE_FEATURES.index("節-機能-原因・理由")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("用言:動")] = 1.0  # 頼み
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("時制:過去")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("動態述語")] = 1.0
    base_phrase_feature_probabilities[1, 5, BASE_PHRASE_FEATURES.index("敬語:丁寧表現")] = 1.0

    dependency_topk = 2
    dependency_logits = torch.zeros(
        (num_examples, max_seq_length, max_seq_length), dtype=torch.float
    )  # (b, word, word)
    dependency_logits[0, 0, 2] = 1.0  # 太郎 -> 次郎
    dependency_logits[0, 1, 0] = 1.0  # と -> 太郎
    dependency_logits[0, 2, 5] = 1.0  # 次郎 -> けんか
    dependency_logits[0, 3, 2] = 1.0  # は -> 次郎
    dependency_logits[0, 4, 5] = 1.0  # よく -> けんか
    # けんか -> [ROOT]
    dependency_logits[0, 5, dataset.examples[0].special_token_indexer.get_morpheme_level_index("[ROOT]")] = 1.0
    dependency_logits[0, 6, 5] = 1.0  # する -> けんか
    dependency_logits[1, 0, 1] = 1.0  # 辛い -> ラーメン
    dependency_logits[1, 1, 3] = 1.0  # ラーメン -> 好きな
    dependency_logits[1, 2, 1] = 1.0  # が -> ラーメン
    dependency_logits[1, 3, 5] = 1.0  # 好きな -> 頼み
    dependency_logits[1, 4, 3] = 1.0  # ので -> 好きな
    # 頼み -> [ROOT]
    dependency_logits[1, 5, dataset.examples[1].special_token_indexer.get_morpheme_level_index("[ROOT]")] = 1.0
    dependency_logits[1, 6, 5] = 1.0  # ました -> 頼み

    dependency_type_logits = torch.zeros(
        (num_examples, max_seq_length, dependency_topk, len(DEPENDENCY_TYPES)), dtype=torch.float
    )
    dependency_type_logits[0, 0, 0, DEPENDENCY_TYPES.index(DepType.PARALLEL)] = 1.0  # 太郎 -> 次郎
    dependency_type_logits[0, 1, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # が -> 太郎
    dependency_type_logits[0, 2, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # 次郎 -> けんか
    dependency_type_logits[0, 3, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # は -> 次郎
    dependency_type_logits[0, 4, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # よく -> けんか
    dependency_type_logits[0, 5, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # けんか -> [ROOT]
    dependency_type_logits[0, 6, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # する -> けんか
    dependency_type_logits[1, 0, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # 辛い -> ラーメン
    dependency_type_logits[1, 1, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # ラーメン -> 好きな
    dependency_type_logits[1, 2, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # が -> ラーメン
    dependency_type_logits[1, 3, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # 好きな -> 頼み
    dependency_type_logits[1, 4, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # ので -> 好きな
    dependency_type_logits[1, 5, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # 頼み -> [ROOT]
    dependency_type_logits[1, 6, 0, DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0  # ました -> 頼み

    # (b, rel, src, tgt)
    flatten_rels = [r for rels in dataset.cohesion_task2rels.values() for r in rels]
    cohesion_logits = torch.zeros((num_examples, len(flatten_rels), max_seq_length, max_seq_length), dtype=torch.float)
    for i in range(num_examples):
        for j, rel in enumerate(flatten_rels):
            if rel == "=":
                k = dataset.examples[i].special_token_indexer.get_morpheme_level_index("[NA]")
            else:
                k = dataset.examples[i].special_token_indexer.get_morpheme_level_index("[NULL]")
            cohesion_logits[i, j, :, k] = 1.0
    cohesion_logits[0, flatten_rels.index("ガ"), 5, 2] = 2.0  # 次郎 ガ けんか
    cohesion_logits[1, flatten_rels.index("ガ"), 0, 1] = 2.0  # ラーメン ガ 辛い
    cohesion_logits[1, flatten_rels.index("ガ"), 3, 1] = 2.0  # ラーメン ガ 好き
    cohesion_logits[
        1, flatten_rels.index("ガ２"), 3, dataset.examples[1].special_token_indexer.get_morpheme_level_index("[著者]")
    ] = 2.0  # 著者 ガ２ 好き
    cohesion_logits[
        1, flatten_rels.index("ガ"), 5, dataset.examples[1].special_token_indexer.get_morpheme_level_index("[著者]")
    ] = 2.0  # 著者 ガ 頼み
    cohesion_logits[1, flatten_rels.index("ヲ"), 5, 1] = 2.0  # ラーメン ヲ 頼み

    # (b, src, tgt, rel)
    discourse_logits = torch.zeros(
        (num_examples, max_seq_length, max_seq_length, len(DISCOURSE_RELATIONS)), dtype=torch.float
    )
    discourse_logits[1, 3, 5, DISCOURSE_RELATIONS.index("原因・理由")] = 1.0  # 好きな - 頼み|原因・理由
    discourse_probabilities = discourse_logits.softmax(dim=3)
    discourse_max_probabilities, discourse_predictions = discourse_probabilities.max(dim=3)

    prediction = {
        "example_ids": torch.arange(num_examples, dtype=torch.long),
        "reading_predictions": reading_logits.argmax(dim=2),
        "reading_subword_map": reading_subword_map,
        "pos_logits": pos_logits,
        "subpos_logits": subpos_logits,
        "conjtype_logits": conjtype_logits,
        "conjform_logits": conjform_logits,
        "word_feature_probabilities": word_feature_probabilities,
        "ne_predictions": ne_predictions,
        "base_phrase_feature_probabilities": base_phrase_feature_probabilities,
        "dependency_predictions": dependency_logits.topk(k=dependency_topk, dim=2).indices,
        "dependency_type_predictions": dependency_type_logits.argmax(dim=3),
        "cohesion_logits": cohesion_logits,
        "discourse_predictions": discourse_predictions,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = WordModuleWriter(AMBIG_SURF_SPECS, destination=tmp_dir / Path("word_prediction.knp"))
        writer.jumandic = build_dummy_jumandic()
        writer.write_on_batch_end(trainer, module, prediction, None, ..., 0, 0)  # type: ignore
        assert isinstance(writer.destination, Path), "destination isn't set"
        assert writer.destination.read_text() == dedent(
            f"""\
            # S-ID:{doc_id_prefix}-0-0 kwja:{version("kwja")}
            * 1P
            + 1P <NE:PERSON:太郎><体言><SM-主体>
            太郎 たろう 太郎 名詞 6 人名 5 * 0 * 0 "代表表記:太郎/たろう 人名" <基本句-主辞>
            と と と 助詞 9 格助詞 1 * 0 * 0
            * 3D
            + 3D <NE:PERSON:次郎><体言><SM-主体>
            次郎 じろう 次郎 名詞 6 人名 5 * 0 * 0 "代表表記:次郎/じろう 人名" <基本句-主辞>
            は は は 助詞 9 副助詞 2 * 0 * 0
            * 3D
            + 3D <修飾>
            よく よく よく 副詞 8 * 0 * 0 * 0 <基本句-主辞><用言表記先頭><用言表記末尾>
            * -1D
            + -1D <rel type="ガ" target="次郎" sid="test-0-0" id="1"/><用言:動><時制:非過去><レベル:C><動態述語><節-区切><節-主辞>
            けんか けんか けんか 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:喧嘩/けんか" <基本句-主辞><用言表記先頭><用言表記末尾><ALT-けんか-けんか-けんか-6-2-0-0-"代表表記:献花/けんか">
            する する する 動詞 2 * 0 サ変動詞 16 基本形 2
            EOS
            # S-ID:{doc_id_prefix}-1-0 kwja:{version("kwja")}
            * 1D
            + 1D <rel type="ガ" target="ラーメン" sid="test-1-0" id="1"/><用言:形><時制:非過去><レベル:B-><状態述語>
            辛い からい 辛い 形容詞 3 * 0 イ形容詞アウオ段 18 基本形 2 <基本句-主辞><用言表記先頭><用言表記末尾>
            * 2D
            + 2D <体言>
            ラーメン らーめん ラーメン 名詞 6 普通名詞 1 * 0 * 0 <基本句-主辞>
            が が が 助詞 9 格助詞 1 * 0 * 0
            * 3D
            + 3D <rel type="ガ" target="ラーメン" sid="test-1-0" id="1"/><rel type="ガ２" target="著者"/><用言:形><時制:非過去><レベル:B+><状態述語><節-機能-原因・理由><節-区切><節-主辞><談話関係:test-1-0/3/原因・理由>
            好きな すきな 好きだ 形容詞 3 * 0 ナ形容詞 21 ダ列基本連体形 3 <基本句-主辞><用言表記先頭><用言表記末尾>
            ので ので のだ 助動詞 5 * 0 ナ形容詞 21 ダ列タ系連用テ形 12
            * -1D
            + -1D <rel type="ガ" target="著者"/><rel type="ヲ" target="ラーメン" sid="test-1-0" id="1"/><用言:動><時制:過去><レベル:C><動態述語><敬語:丁寧表現><節-区切><節-主辞>
            頼み たのみ 頼む 動詞 2 * 0 子音動詞マ行 9 基本連用形 8 <基本句-主辞><用言表記先頭><用言表記末尾>
            ました ました ます 接尾辞 14 動詞性接尾辞 7 動詞性接尾辞ます型 31 タ形 7
            EOS
            """
        )
        juman_file.close()
