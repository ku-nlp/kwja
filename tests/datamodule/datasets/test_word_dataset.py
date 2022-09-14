import json
import textwrap
from pathlib import Path

from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.props import DepType
from transformers.utils import PaddingStrategy

from jula.datamodule.datasets.word_dataset import WordDataset, WordExampleSet
from jula.datamodule.examples import (
    BasePhraseFeatureExample,
    CohesionExample,
    CohesionTask,
    DependencyExample,
    DiscourseExample,
    ReadingExample,
    WordFeatureExample,
)
from jula.datamodule.extractors import PasAnnotation
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TYPES,
    CONJTYPE_TYPES,
    DEPENDENCY_TYPE2INDEX,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    POS_TYPES,
    SUBPOS_TYPES,
    WORD_FEATURES,
)

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")
data_dir = here.parent.parent / "data"

reading_resource_path = here / "reading_files"

exophora_referents = ["著者", "読者", "不特定:人", "不特定:物"]
special_tokens = exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]
word_dataset_kwargs = dict(
    pas_cases=ListConfig(["ガ", "ヲ", "ニ", "ガ２"]),
    bar_rels=ListConfig(["ノ"]),
    exophora_referents=ListConfig(exophora_referents),
    cohesion_tasks=ListConfig(["pas_analysis", "bridging", "coreference"]),
    special_tokens=ListConfig(special_tokens),
    restrict_cohesion_target=True,
    tokenizer_kwargs={"additional_special_tokens": special_tokens},
    reading_resource_path=str(reading_resource_path),
    document_split_stride=1,
)


def test_init():
    _ = WordDataset(str(path), **word_dataset_kwargs)


def test_getitem():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length, **word_dataset_kwargs)
    for i in range(len(dataset)):
        document = dataset.documents[i]
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "subword_map" in item
        assert "reading_subword_map" in item
        assert "reading_ids" in item
        assert "mrph_types" in item
        assert "word_features" in item
        assert "base_phrase_features" in item
        assert "dependencies" in item
        assert "intra_mask" in item
        assert "dependency_types" in item
        assert "discourse_relations" in item
        assert "cohesion_target" in item
        assert "cohesion_mask" in item
        assert "tokens" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["subword_map"].shape == (max_seq_length, max_seq_length)
        assert (item["subword_map"].sum(dim=1) != 0).sum() == len(document.morphemes) + dataset.num_special_tokens
        assert item["reading_subword_map"].shape == (max_seq_length, max_seq_length)
        assert (item["reading_subword_map"].sum(dim=1) != 0).sum() == len(document.morphemes)
        assert item["reading_ids"].shape == (max_seq_length,)
        assert item["mrph_types"].shape == (max_seq_length, 4)
        assert item["word_features"].shape == (max_seq_length, len(WORD_FEATURES))
        assert item["base_phrase_features"].shape == (max_seq_length, len(BASE_PHRASE_FEATURES))
        assert item["dependencies"].shape == (max_seq_length,)
        assert item["intra_mask"].shape == (max_seq_length, max_seq_length)
        assert item["dependency_types"].shape == (max_seq_length,)
        assert item["cohesion_target"].shape == (6, max_seq_length, max_seq_length)
        assert item["cohesion_mask"].shape == (6, max_seq_length, max_seq_length)
        assert item["discourse_relations"].shape == (max_seq_length, max_seq_length)


def test_encode():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length, **word_dataset_kwargs)
    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:000-1 KNP:5.0-2ad4f6df
            * 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ>
            + 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ><解析格:ガ>
            風 かぜ 風 名詞 6 普通名詞 1 * 0 * 0 "代表表記:風/かぜ カテゴリ:抽象物 漢字読み:訓" <代表表記:風/かぜ><カテゴリ:抽象物><漢字読み:訓><正規化代表表記:風/かぜ><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始><文節主辞>
            が が が 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく>
            + -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく><用言代表表記:吹く/ふく><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:ガ:風><格解析結果:吹く/ふく:動1:ガ/C/風/0/0/000-1;ニ/U/-/-/-/-;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;時間/U/-/-/-/-><標準用言代表表記:吹く/ふく><談話関係:000-2/2/条件(順方向)>
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
    encoding = dataset.tokenizer(
        [morpheme.text for morpheme in document.morphemes],
        is_split_into_words=True,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=False,
        max_length=dataset.max_seq_length - dataset.num_special_tokens,
    ).encodings[0]
    reading_example = ReadingExample()
    reading_example.load(document, dataset.reading_aligner)
    word_feature_example = WordFeatureExample()
    word_feature_example.load(document)
    base_phrase_feature_example = BasePhraseFeatureExample()
    base_phrase_feature_example.load(document)
    cohesion_example = CohesionExample()
    cohesion_example.load(document, dataset.cohesion_tasks, dataset.extractors)
    dependency_example = DependencyExample()
    dependency_example.load(document)
    discourse_example = DiscourseExample()
    discourse_example.load(document)
    example = WordExampleSet(
        example_id=0,
        doc_id="",
        encoding=encoding,
        reading_example=reading_example,
        word_feature_example=word_feature_example,
        base_phrase_feature_example=base_phrase_feature_example,
        dependency_example=dependency_example,
        cohesion_example=cohesion_example,
        discourse_example=discourse_example,
    )
    features = dataset.encode(example)

    reading_ids = [IGNORE_INDEX for _ in range(max_seq_length)]
    # reading_ids[0]: CLS
    reading_ids[1] = dataset.reading2id["かぜ"]  # 風
    reading_ids[2] = dataset.reading2id["[ID]"]  # が
    reading_ids[3] = dataset.reading2id["ふく"]  # 吹く
    reading_ids[4] = dataset.reading2id["[ID]"]  # 。
    reading_ids[5] = dataset.reading2id["[ID]"]  # する
    reading_ids[6] = dataset.reading2id["[ID]"]  # と
    reading_ids[7] = dataset.reading2id["[ID]"]  # _
    reading_ids[8] = dataset.reading2id["おけ"]  # 桶
    reading_ids[9] = dataset.reading2id["や"]  # 屋
    reading_ids[10] = dataset.reading2id["[ID]"]  # が
    reading_ids[11] = dataset.reading2id["[ID]"]  # _
    reading_ids[12] = dataset.reading2id["もう"]  # 儲
    reading_ids[13] = dataset.reading2id["[ID]"]  # か
    reading_ids[14] = dataset.reading2id["[ID]"]  # る
    reading_ids[15] = dataset.reading2id["[ID]"]  # 。

    mrph_types = [[IGNORE_INDEX] * 4 for _ in range(max_seq_length)]
    # 0: 風
    mrph_types[0][0] = POS_TYPES.index("名詞")
    mrph_types[0][1] = SUBPOS_TYPES.index("普通名詞")
    mrph_types[0][2] = CONJTYPE_TYPES.index("*")
    mrph_types[0][3] = CONJFORM_TYPES.index("*")
    # 1: が
    mrph_types[1][0] = POS_TYPES.index("助詞")
    mrph_types[1][1] = SUBPOS_TYPES.index("格助詞")
    mrph_types[1][2] = CONJTYPE_TYPES.index("*")
    mrph_types[1][3] = CONJFORM_TYPES.index("*")
    # 2: 吹く
    mrph_types[2][0] = POS_TYPES.index("動詞")
    mrph_types[2][1] = SUBPOS_TYPES.index("*")
    mrph_types[2][2] = CONJTYPE_TYPES.index("子音動詞カ行")
    mrph_types[2][3] = CONJFORM_TYPES.index("基本形")
    # 3: 。
    mrph_types[3][0] = POS_TYPES.index("特殊")
    mrph_types[3][1] = SUBPOS_TYPES.index("句点")
    mrph_types[3][2] = CONJTYPE_TYPES.index("*")
    mrph_types[3][3] = CONJFORM_TYPES.index("*")
    # 4: すると
    mrph_types[4][0] = POS_TYPES.index("接続詞")
    mrph_types[4][1] = SUBPOS_TYPES.index("*")
    mrph_types[4][2] = CONJTYPE_TYPES.index("*")
    mrph_types[4][3] = CONJFORM_TYPES.index("*")
    # 5: 桶屋
    mrph_types[5][0] = POS_TYPES.index("名詞")
    mrph_types[5][1] = SUBPOS_TYPES.index("普通名詞")
    mrph_types[5][2] = CONJTYPE_TYPES.index("*")
    mrph_types[5][3] = CONJFORM_TYPES.index("*")
    # 6: が
    mrph_types[6][0] = POS_TYPES.index("助詞")
    mrph_types[6][1] = SUBPOS_TYPES.index("格助詞")
    mrph_types[6][2] = CONJTYPE_TYPES.index("*")
    mrph_types[6][3] = CONJFORM_TYPES.index("*")
    # 7: 儲かる
    mrph_types[7][0] = POS_TYPES.index("動詞")
    mrph_types[7][1] = SUBPOS_TYPES.index("*")
    mrph_types[7][2] = CONJTYPE_TYPES.index("子音動詞ラ行")
    mrph_types[7][3] = CONJFORM_TYPES.index("基本形")
    # 8: 。
    mrph_types[8][0] = POS_TYPES.index("特殊")
    mrph_types[8][1] = SUBPOS_TYPES.index("句点")
    mrph_types[8][2] = CONJTYPE_TYPES.index("*")
    mrph_types[8][3] = CONJFORM_TYPES.index("*")
    assert features["mrph_types"].tolist() == mrph_types

    word_features = [[0] * len(WORD_FEATURES) for _ in range(max_seq_length)]
    # 0: 風
    word_features[0][WORD_FEATURES.index("基本句-主辞")] = 1
    # 1: が
    word_features[1][WORD_FEATURES.index("基本句-区切")] = 1
    word_features[1][WORD_FEATURES.index("文節-区切")] = 1
    # 2: 吹く
    word_features[2][WORD_FEATURES.index("基本句-主辞")] = 1
    word_features[2][WORD_FEATURES.index("用言表記先頭")] = 1
    word_features[2][WORD_FEATURES.index("用言表記末尾")] = 1
    # 3: 。
    word_features[3][WORD_FEATURES.index("基本句-区切")] = 1
    word_features[3][WORD_FEATURES.index("文節-区切")] = 1
    # 4: すると
    word_features[4][WORD_FEATURES.index("基本句-主辞")] = 1
    word_features[4][WORD_FEATURES.index("基本句-区切")] = 1
    word_features[4][WORD_FEATURES.index("文節-区切")] = 1
    # 5: 桶屋
    word_features[5][WORD_FEATURES.index("基本句-主辞")] = 1
    # 6: が
    word_features[6][WORD_FEATURES.index("基本句-区切")] = 1
    word_features[6][WORD_FEATURES.index("文節-区切")] = 1
    # 7: 儲かる
    word_features[7][WORD_FEATURES.index("基本句-主辞")] = 1
    word_features[2][WORD_FEATURES.index("用言表記先頭")] = 1
    word_features[2][WORD_FEATURES.index("用言表記末尾")] = 1
    # 8: 。
    word_features[8][WORD_FEATURES.index("基本句-区切")] = 1
    word_features[8][WORD_FEATURES.index("文節-区切")] = 1

    base_phrase_head_indices = {base_phrase.head.global_index for base_phrase in document.base_phrases}
    base_phrase_features = [
        [0] * len(BASE_PHRASE_FEATURES)
        if morpheme_global_index in base_phrase_head_indices
        else [IGNORE_INDEX] * len(BASE_PHRASE_FEATURES)
        for morpheme_global_index in range(max_seq_length)
    ]
    # 0: 風
    base_phrase_features[0][BASE_PHRASE_FEATURES.index("体言")] = 1
    # 2: 吹く
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("用言:動")] = 1
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("動態述語")] = 1
    # 4: すると
    base_phrase_features[4][BASE_PHRASE_FEATURES.index("修飾")] = 1
    base_phrase_features[4][BASE_PHRASE_FEATURES.index("節-前向き機能-条件")] = 1
    # 5: 桶屋
    base_phrase_features[5][BASE_PHRASE_FEATURES.index("体言")] = 1
    # 7: 儲かる
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("用言:動")] = 1
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("節-区切")] = 1
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("節-主辞")] = 1
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("レベル:C")] = 1
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("動態述語")] = 1
    assert features["base_phrase_features"].tolist() == base_phrase_features

    dependencies = [IGNORE_INDEX for _ in range(max_seq_length)]
    # 0: 風 -> 2: 吹く
    dependencies[0] = 2
    # 1: が -> 0: 風
    dependencies[1] = 0
    # 2: 吹く -> ROOT
    dependencies[2] = max_seq_length - 1
    # 3: 。 -> 2: 吹く
    dependencies[3] = 2
    # 4: すると -> 7: 儲かる
    dependencies[4] = 7
    # 5: 桶屋 -> 7: 儲かる
    dependencies[5] = 7
    # 6: が -> 5: 桶屋
    dependencies[6] = 5
    # 7: 儲かる -> ROOT
    dependencies[7] = max_seq_length - 1
    # 8: 。 -> 7: 儲かる
    dependencies[8] = 7
    assert features["dependencies"].tolist() == dependencies

    dependency_types = [IGNORE_INDEX for _ in range(max_seq_length)]
    # 0: 風 -> 2: 吹く
    dependency_types[0] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 1: が -> 0: 風
    dependency_types[1] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 2: 吹く -> ROOT ("D"として扱う)
    dependency_types[2] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 3: 。 -> 2: 吹く
    dependency_types[3] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 4: すると -> 7: 儲かる
    dependency_types[4] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 5: 桶屋 -> 7: 儲かる
    dependency_types[5] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 6: が -> 5: 桶屋
    dependency_types[6] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 7: 儲かる -> ROOT ("D"として扱う)
    dependency_types[7] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    # 8: 。 -> 7: 儲かる
    dependency_types[8] = DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]
    assert features["dependency_types"].tolist() == dependency_types

    discourse_relations = [[IGNORE_INDEX] * max_seq_length for _ in range(max_seq_length)]
    discourse_relations[2][2] = DISCOURSE_RELATIONS.index("談話関係なし")
    discourse_relations[2][7] = DISCOURSE_RELATIONS.index("条件")
    discourse_relations[7][2] = DISCOURSE_RELATIONS.index("談話関係なし")
    discourse_relations[7][7] = DISCOURSE_RELATIONS.index("談話関係なし")
    assert features["discourse_relations"].tolist() == discourse_relations


def test_pas():
    max_seq_length = 512
    dataset = WordDataset(str(data_dir / "knp"), max_seq_length=max_seq_length, **word_dataset_kwargs)
    example = [e for e in dataset.examples if e.doc_id == "w201106-0000060560"][0].cohesion_example
    example_expected = json.loads((data_dir / "expected/example/0.json").read_text())
    mrphs_exp = example_expected["mrphs"]
    annotation: PasAnnotation = example.annotations[CohesionTask.PAS_ANALYSIS]
    phrases = example.phrases[CohesionTask.PAS_ANALYSIS]
    mrphs = example.mrphs[CohesionTask.PAS_ANALYSIS]

    assert len(mrphs) == len(mrphs_exp)
    for phrase in phrases:
        arguments: dict[str, list[str]] = annotation.arguments_set[phrase.dtid]
        for case in dataset.pas_cases:
            arg_strings = [arg[:-2] if arg[-2:] in ("%C", "%N", "%O") else arg for arg in arguments[case]]
            arg_strings = [(s if s in dataset.special_to_index else str(phrases[int(s)].dmid)) for s in arg_strings]
            assert set(arg_strings) == set(mrphs_exp[phrase.dmid]["arguments"][case])
        for dmid in phrase.dmids:
            mrph = mrphs[dmid]
            mrph_exp = mrphs_exp[dmid]
            assert mrph.surf == mrph_exp["surf"]
            if mrph.is_target or example.mrphs[CohesionTask.BRIDGING][dmid].is_target:
                candidates = set(phrases[i].dmid for i in phrase.candidates)
                bar_phrases = example.phrases[CohesionTask.BRIDGING]
                candidates |= set(bar_phrases[i].dmid for i in bar_phrases[phrase.dtid].candidates)
                assert candidates == set(mrph_exp["arg_candidates"])
