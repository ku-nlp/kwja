import textwrap
from pathlib import Path

import torch
from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.props import DepType
from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.word_dataset import WordDataset
from kwja.datamodule.examples import WordExample
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
)

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")
data_dir = here.parent.parent / "data"

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
    _ = WordDataset(str(path), tokenizer, **word_dataset_args)


def test_getitem():
    max_seq_length = 256
    dataset = WordDataset(str(path), tokenizer, **{**word_dataset_args, "max_seq_length": max_seq_length})
    num_cohesion_rels = len([r for utils in dataset.cohesion_task2utils.values() for r in utils.rels])
    for i in range(len(dataset)):
        document = dataset.documents[i]
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target_mask" in item
        assert "subword_map" in item
        assert "reading_labels" in item
        assert "reading_subword_map" in item
        assert "pos_labels" in item
        assert "subpos_labels" in item
        assert "conjtype_labels" in item
        assert "conjform_labels" in item
        assert "word_feature_labels" in item
        assert "ne_labels" in item
        assert "base_phrase_feature_labels" in item
        assert "dependency_labels" in item
        assert "dependency_mask" in item
        assert "dependency_type_labels" in item
        assert "cohesion_labels" in item
        assert "cohesion_mask" in item
        assert "discourse_labels" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["target_mask"].shape == (max_seq_length,)
        assert item["subword_map"].shape == (max_seq_length, max_seq_length)
        assert (item["subword_map"].sum(dim=1) != 0).sum() == len(document.morphemes) + dataset.num_special_tokens
        assert item["reading_labels"].shape == (max_seq_length,)
        assert item["reading_subword_map"].shape == (max_seq_length, max_seq_length)
        assert (item["reading_subword_map"].sum(dim=1) != 0).sum() == len(document.morphemes)
        assert item["pos_labels"].shape == (max_seq_length,)
        assert item["subpos_labels"].shape == (max_seq_length,)
        assert item["conjtype_labels"].shape == (max_seq_length,)
        assert item["conjform_labels"].shape == (max_seq_length,)
        assert item["word_feature_labels"].shape == (max_seq_length, len(WORD_FEATURES))
        assert item["ne_labels"].shape == (max_seq_length,)
        assert item["base_phrase_feature_labels"].shape == (max_seq_length, len(BASE_PHRASE_FEATURES))
        assert item["dependency_labels"].shape == (max_seq_length,)
        assert item["dependency_mask"].shape == (max_seq_length, max_seq_length)
        assert item["dependency_type_labels"].shape == (max_seq_length,)
        assert item["cohesion_labels"].shape == (num_cohesion_rels, max_seq_length, max_seq_length)
        assert item["cohesion_mask"].shape == (num_cohesion_rels, max_seq_length, max_seq_length)
        assert item["discourse_labels"].shape == (max_seq_length, max_seq_length)


def test_encode():
    max_seq_length = 24
    dataset = WordDataset(str(path), tokenizer, **{**word_dataset_args, "max_seq_length": max_seq_length})
    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:000-1 KNP:5.0-2ad4f6df
            * 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ>
            + 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ><解析格:ガ>
            風 かぜ 風 名詞 6 普通名詞 1 * 0 * 0 "代表表記:風/かぜ カテゴリ:抽象物 漢字読み:訓" <代表表記:風/かぜ><カテゴリ:抽象物><漢字読み:訓><正規化代表表記:風/かぜ><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始><文節主辞>
            が が が 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく>
            + -1D <rel type="ニ" target="不特定:人"/><BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく><用言代表表記:吹く/ふく><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:ガ:風><格解析結果:吹く/ふく:動1:ガ/C/風/0/0/000-1;ニ/U/-/-/-/-;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;時間/U/-/-/-/-><標準用言代表表記:吹く/ふく><談話関係:000-2/2/条件(順方向)>
            吹く ふく 吹く 動詞 2 * 0 子音動詞カ行 2 基本形 2 "代表表記:吹く/ふく 補文ト" <代表表記:吹く/ふく><補文ト><正規化代表表記:吹く/ふく><かな漢字><活用語><表現文末><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            。 。 。 特殊 1 句点 1 * 0 * 0 NIL <英記号><記号><文末><付属>
            EOS
            # S-ID:000-2 KNP:5.0-2ad4f6df
            * 2D <BGH:すると/すると><文頭><接続詞><修飾><係:連用><区切:0-4><連用要素><連用節><正規化代表表記:すると/すると><主辞代表表記:すると/すると>
            + 2D <BGH:すると/すると><文頭><接続詞><修飾><係:連用><区切:0-4><連用要素><連用節><節-前向き機能-条件:すると><正規化代表表記:すると/すると><主辞代表表記:すると/すると><解析格:修飾>
            すると すると すると 接続詞 10 * 0 * 0 * 0 "代表表記:すると/すると" <代表表記:すると/すると><正規化代表表記:すると/すると><かな漢字><ひらがな><文頭><自立><内容語><タグ単位始><文節始><文節主辞>
            * 2D <ガ><助詞><体言><係:ガ格><区切:0-0><格要素><連用要素><正規化代表表記:桶屋/桶屋><主辞代表表記:桶屋/桶屋>
            + 2D <ガ><助詞><体言><係:ガ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:桶屋/桶屋><主辞代表表記:桶屋/桶屋><Wikipediaリダイレクト:桶><解析格:ガ>
            桶屋 桶屋 桶屋 名詞 6 普通名詞 1 * 0 * 0 "自動獲得:Wikipedia 読み不明 Wikipediaリダイレクト:桶 疑似代表表記 代表表記:桶屋/桶屋" <自動獲得:Wikipedia><読み不明><Wikipediaリダイレクト:桶><疑似代表表記><代表表記:桶屋/桶屋><正規化代表表記:桶屋/桶屋><漢字><かな漢字><名詞相当語><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            が が が 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * -1D <BGH:儲かる/もうかる><文末><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:儲かる/もうかる><主辞代表表記:儲かる/もうかる>
            + -1D <BGH:儲かる/もうかる><文末><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:儲かる/もうかる><主辞代表表記:儲かる/もうかる><用言代表表記:儲かる/もうかる><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:修飾:すると><格関係1:ガ:桶屋><格解析結果:儲かる/もうかる:動2:ガ/C/桶屋/1/0/000-2;修飾/C/すると/0/0/000-2><標準用言代表表記:儲かる/もうかる>
            儲かる もうかる 儲かる 動詞 2 * 0 子音動詞ラ行 10 基本形 2 "代表表記:儲かる/もうかる ドメイン:ビジネス 自他動詞:他:儲ける/もうける" <代表表記:儲かる/もうかる><ドメイン:ビジネス><自他動詞:他:儲ける/もうける><正規化代表表記:儲かる/もうかる><かな漢字><活用語><表現文末><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            。 。 。 特殊 1 句点 1 * 0 * 0 NIL <英記号><記号><文末><付属>
            EOS
            """
        )
    )
    tokenizer_input = [m.text for m in document.morphemes]
    if dataset.tokenizer_input_format == "text":
        tokenizer_input = " ".join(tokenizer_input)
    encoding = dataset.tokenizer(
        tokenizer_input,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=False,
        max_length=dataset.max_seq_length - dataset.num_special_tokens,
        is_split_into_words=dataset.tokenizer_input_format == "words",
    ).encodings[0]
    word_example = WordExample(0, encoding=encoding)
    word_example.load_document(document, dataset.reading_aligner, dataset.cohesion_task2utils)
    word_example.load_discourse_document(document)
    features = dataset.encode(word_example)

    reading_labels = [IGNORE_INDEX for _ in range(max_seq_length)]
    # reading_labels[0]: CLS
    reading_labels[1] = dataset.reading2reading_id["かぜ"]  # 風
    reading_labels[2] = dataset.reading2reading_id["[ID]"]  # が
    reading_labels[3] = dataset.reading2reading_id["ふく"]  # 吹く
    reading_labels[4] = dataset.reading2reading_id["[ID]"]  # 。
    reading_labels[5] = dataset.reading2reading_id["[ID]"]  # する
    reading_labels[6] = dataset.reading2reading_id["[ID]"]  # と
    reading_labels[7] = dataset.reading2reading_id["[ID]"]  # _
    reading_labels[8] = dataset.reading2reading_id["おけ"]  # 桶
    reading_labels[9] = dataset.reading2reading_id["や"]  # 屋
    reading_labels[10] = dataset.reading2reading_id["[ID]"]  # が
    reading_labels[11] = dataset.reading2reading_id["[ID]"]  # _
    reading_labels[12] = dataset.reading2reading_id["もう"]  # 儲
    reading_labels[13] = dataset.reading2reading_id["[ID]"]  # か
    reading_labels[14] = dataset.reading2reading_id["[ID]"]  # る
    reading_labels[15] = dataset.reading2reading_id["[ID]"]  # 。

    morpheme_attribute_labels = torch.tensor([[IGNORE_INDEX] * 4 for _ in range(max_seq_length)], dtype=torch.long)
    # 0: 風
    morpheme_attribute_labels[0][0] = POS_TAGS.index("名詞")
    morpheme_attribute_labels[0][1] = SUBPOS_TAGS.index("普通名詞")
    morpheme_attribute_labels[0][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[0][3] = CONJFORM_TAGS.index("*")
    # 1: が
    morpheme_attribute_labels[1][0] = POS_TAGS.index("助詞")
    morpheme_attribute_labels[1][1] = SUBPOS_TAGS.index("格助詞")
    morpheme_attribute_labels[1][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[1][3] = CONJFORM_TAGS.index("*")
    # 2: 吹く
    morpheme_attribute_labels[2][0] = POS_TAGS.index("動詞")
    morpheme_attribute_labels[2][1] = SUBPOS_TAGS.index("*")
    morpheme_attribute_labels[2][2] = CONJTYPE_TAGS.index("子音動詞カ行")
    morpheme_attribute_labels[2][3] = CONJFORM_TAGS.index("基本形")
    # 3: 。
    morpheme_attribute_labels[3][0] = POS_TAGS.index("特殊")
    morpheme_attribute_labels[3][1] = SUBPOS_TAGS.index("句点")
    morpheme_attribute_labels[3][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[3][3] = CONJFORM_TAGS.index("*")
    # 4: すると
    morpheme_attribute_labels[4][0] = POS_TAGS.index("接続詞")
    morpheme_attribute_labels[4][1] = SUBPOS_TAGS.index("*")
    morpheme_attribute_labels[4][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[4][3] = CONJFORM_TAGS.index("*")
    # 5: 桶屋
    morpheme_attribute_labels[5][0] = POS_TAGS.index("名詞")
    morpheme_attribute_labels[5][1] = SUBPOS_TAGS.index("普通名詞")
    morpheme_attribute_labels[5][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[5][3] = CONJFORM_TAGS.index("*")
    # 6: が
    morpheme_attribute_labels[6][0] = POS_TAGS.index("助詞")
    morpheme_attribute_labels[6][1] = SUBPOS_TAGS.index("格助詞")
    morpheme_attribute_labels[6][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[6][3] = CONJFORM_TAGS.index("*")
    # 7: 儲かる
    morpheme_attribute_labels[7][0] = POS_TAGS.index("動詞")
    morpheme_attribute_labels[7][1] = SUBPOS_TAGS.index("*")
    morpheme_attribute_labels[7][2] = CONJTYPE_TAGS.index("子音動詞ラ行")
    morpheme_attribute_labels[7][3] = CONJFORM_TAGS.index("基本形")
    # 8: 。
    morpheme_attribute_labels[8][0] = POS_TAGS.index("特殊")
    morpheme_attribute_labels[8][1] = SUBPOS_TAGS.index("句点")
    morpheme_attribute_labels[8][2] = CONJTYPE_TAGS.index("*")
    morpheme_attribute_labels[8][3] = CONJFORM_TAGS.index("*")
    assert features["pos_labels"].tolist() == morpheme_attribute_labels[:, 0].tolist()
    assert features["subpos_labels"].tolist() == morpheme_attribute_labels[:, 1].tolist()
    assert features["conjtype_labels"].tolist() == morpheme_attribute_labels[:, 2].tolist()
    assert features["conjform_labels"].tolist() == morpheme_attribute_labels[:, 3].tolist()

    word_feature_labels = [[0] * len(WORD_FEATURES) for _ in range(max_seq_length)]
    # 0: 風
    word_feature_labels[0][WORD_FEATURES.index("基本句-主辞")] = 1
    # 1: が
    word_feature_labels[1][WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[1][WORD_FEATURES.index("文節-区切")] = 1
    # 2: 吹く
    word_feature_labels[2][WORD_FEATURES.index("基本句-主辞")] = 1
    word_feature_labels[2][WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[2][WORD_FEATURES.index("用言表記末尾")] = 1
    # 3: 。
    word_feature_labels[3][WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[3][WORD_FEATURES.index("文節-区切")] = 1
    # 4: すると
    word_feature_labels[4][WORD_FEATURES.index("基本句-主辞")] = 1
    word_feature_labels[4][WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[4][WORD_FEATURES.index("文節-区切")] = 1
    # 5: 桶屋
    word_feature_labels[5][WORD_FEATURES.index("基本句-主辞")] = 1
    # 6: が
    word_feature_labels[6][WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[6][WORD_FEATURES.index("文節-区切")] = 1
    # 7: 儲かる
    word_feature_labels[7][WORD_FEATURES.index("基本句-主辞")] = 1
    word_feature_labels[2][WORD_FEATURES.index("用言表記先頭")] = 1
    word_feature_labels[2][WORD_FEATURES.index("用言表記末尾")] = 1
    # 8: 。
    word_feature_labels[8][WORD_FEATURES.index("基本句-区切")] = 1
    word_feature_labels[8][WORD_FEATURES.index("文節-区切")] = 1

    base_phrase_head_indices = {base_phrase.head.global_index for base_phrase in document.base_phrases}
    base_phrase_feature_labels = [
        [0] * len(BASE_PHRASE_FEATURES)
        if morpheme_global_index in base_phrase_head_indices
        else [IGNORE_INDEX] * len(BASE_PHRASE_FEATURES)
        for morpheme_global_index in range(max_seq_length)
    ]
    # 0: 風
    base_phrase_feature_labels[0][BASE_PHRASE_FEATURES.index("体言")] = 1
    # 2: 吹く
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("用言:動")] = 1
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[2][BASE_PHRASE_FEATURES.index("動態述語")] = 1
    # 4: すると
    base_phrase_feature_labels[4][BASE_PHRASE_FEATURES.index("修飾")] = 1
    base_phrase_feature_labels[4][BASE_PHRASE_FEATURES.index("節-前向き機能-条件")] = 1
    # 5: 桶屋
    base_phrase_feature_labels[5][BASE_PHRASE_FEATURES.index("体言")] = 1
    # 7: 儲かる
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("用言:動")] = 1
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_feature_labels[7][BASE_PHRASE_FEATURES.index("動態述語")] = 1
    assert features["base_phrase_feature_labels"].tolist() == base_phrase_feature_labels

    dependency_labels = [IGNORE_INDEX for _ in range(max_seq_length)]
    # 0: 風 -> 2: 吹く
    dependency_labels[0] = 2
    # 1: が -> 0: 風
    dependency_labels[1] = 0
    # 2: 吹く -> ROOT
    dependency_labels[2] = max_seq_length - 1
    # 3: 。 -> 2: 吹く
    dependency_labels[3] = 2
    # 4: すると -> 7: 儲かる
    dependency_labels[4] = 7
    # 5: 桶屋 -> 7: 儲かる
    dependency_labels[5] = 7
    # 6: が -> 5: 桶屋
    dependency_labels[6] = 5
    # 7: 儲かる -> ROOT
    dependency_labels[7] = max_seq_length - 1
    # 8: 。 -> 7: 儲かる
    dependency_labels[8] = 7
    assert features["dependency_labels"].tolist() == dependency_labels

    dependency_type_labels = [IGNORE_INDEX for _ in range(max_seq_length)]
    # 0: 風 -> 2: 吹く
    dependency_type_labels[0] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 1: が -> 0: 風
    dependency_type_labels[1] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 2: 吹く -> ROOT ("D"として扱う)
    dependency_type_labels[2] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 3: 。 -> 2: 吹く
    dependency_type_labels[3] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 4: すると -> 7: 儲かる
    dependency_type_labels[4] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 5: 桶屋 -> 7: 儲かる
    dependency_type_labels[5] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 6: が -> 5: 桶屋
    dependency_type_labels[6] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 7: 儲かる -> ROOT ("D"として扱う)
    dependency_type_labels[7] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    # 8: 。 -> 7: 儲かる
    dependency_type_labels[8] = DEPENDENCY_TYPES.index(DepType.DEPENDENCY)
    assert features["dependency_type_labels"].tolist() == dependency_type_labels

    cohesion_labels = []
    for cohesion_task, cohesion_utils in dataset.cohesion_task2utils.items():
        for rel in cohesion_utils.rels:
            rel_labels = [[0] * max_seq_length for _ in range(max_seq_length)]
            if rel == "ガ":
                # 吹く -> 風, 儲かる -> 桶屋
                rel_labels[2][0] = 1
                rel_labels[7][5] = 1
            elif rel in {"ヲ", "ガ２"}:
                rel_labels[2][dataset.special_token2index["[NULL]"]] = 1
                rel_labels[7][dataset.special_token2index["[NULL]"]] = 1
            elif rel == "ニ":
                rel_labels[2][dataset.special_token2index["不特定:人"]] = 1
                rel_labels[7][dataset.special_token2index["[NULL]"]] = 1
            elif rel == "ノ":
                # 風, 桶
                rel_labels[0][dataset.special_token2index["[NULL]"]] = 1
                rel_labels[5][dataset.special_token2index["[NULL]"]] = 1
            elif rel == "=":
                # 風, 桶
                rel_labels[0][dataset.special_token2index["[NA]"]] = 1
                rel_labels[5][dataset.special_token2index["[NA]"]] = 1
            cohesion_labels.append(rel_labels)
    assert features["cohesion_labels"].tolist() == cohesion_labels

    discourse_labels = [[IGNORE_INDEX] * max_seq_length for _ in range(max_seq_length)]
    discourse_labels[2][2] = DISCOURSE_RELATIONS.index("談話関係なし")
    discourse_labels[2][7] = DISCOURSE_RELATIONS.index("条件")
    discourse_labels[7][2] = DISCOURSE_RELATIONS.index("談話関係なし")
    discourse_labels[7][7] = DISCOURSE_RELATIONS.index("談話関係なし")
    assert features["discourse_labels"].tolist() == discourse_labels
