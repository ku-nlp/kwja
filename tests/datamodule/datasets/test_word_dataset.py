import textwrap
from pathlib import Path

from rhoknp import Document

from jula.datamodule.datasets.word_dataset import WordDataset
from jula.utils.features import BASE_PHRASE_FEATURES, WORD_FEATURES

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = WordDataset(str(path))


def test_getitem():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        document = dataset.documents[i]
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "subword_map" in item
        assert "word_features" in item
        assert "base_phrase_features" in item
        assert "dependencies" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["subword_map"].shape == (max_seq_length, max_seq_length)
        assert (item["subword_map"].sum(dim=1) != 0).sum() == len(document.morphemes)
        assert item["base_phrase_features"].shape == (
            max_seq_length,
            len(BASE_PHRASE_FEATURES),
        )
        assert item["word_features"].shape == (max_seq_length, len(WORD_FEATURES))
        assert item["dependencies"].shape == (max_seq_length, max_seq_length)


def test_encode():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length)
    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:000-1 KNP:5.0-2ad4f6df
            * 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ>
            + 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ><解析格:ガ>
            風 かぜ 風 名詞 6 普通名詞 1 * 0 * 0 "代表表記:風/かぜ カテゴリ:抽象物 漢字読み:訓" <代表表記:風/かぜ><カテゴリ:抽象物><漢字読み:訓><正規化代表表記:風/かぜ><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始><文節主辞>
            が が が 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく>
            + -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく><用言代表表記:吹く/ふく><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:ガ:風><格解析結果:吹く/ふく:動1:ガ/C/風/0/0/1;ニ/U/-/-/-/-;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;時間/U/-/-/-/-><標準用言代表表記:吹く/ふく>
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
            + -1D <BGH:儲かる/もうかる><文末><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:儲かる/もうかる><主辞代表表記:儲かる/もうかる><用言代表表記:儲かる/もうかる><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:修飾:すると><格関係1:ガ:桶屋><格解析結果:儲かる/もうかる:動2:ガ/C/桶屋/1/0/2;修飾/C/すると/0/0/2><標準用言代表表記:儲かる/もうかる>
            儲かる もうかる 儲かる 動詞 2 * 0 子音動詞ラ行 10 基本形 2 "代表表記:儲かる/もうかる ドメイン:ビジネス 自他動詞:他:儲ける/もうける" <代表表記:儲かる/もうかる><ドメイン:ビジネス><自他動詞:他:儲ける/もうける><正規化代表表記:儲かる/もうかる><かな漢字><活用語><表現文末><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            。 。 。 特殊 1 句点 1 * 0 * 0 NIL <英記号><記号><文末><付属>
            EOS
            """
        )
    )
    encoding = dataset.encode(document)

    word_features = [[0.0] * len(WORD_FEATURES) for _ in range(max_seq_length)]
    # 0: 風
    word_features[0][WORD_FEATURES.index("基本句-主辞")] = 1.0
    # 1: が
    word_features[1][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_features[1][WORD_FEATURES.index("文節-区切")] = 1.0
    # 2: 吹く
    word_features[2][WORD_FEATURES.index("基本句-主辞")] = 1.0
    # 3: 。
    word_features[3][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_features[3][WORD_FEATURES.index("文節-区切")] = 1.0
    # 4: すると
    word_features[4][WORD_FEATURES.index("基本句-主辞")] = 1.0
    word_features[4][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_features[4][WORD_FEATURES.index("文節-区切")] = 1.0
    # 5: 桶屋
    word_features[5][WORD_FEATURES.index("基本句-主辞")] = 1.0
    # 6: が
    word_features[6][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_features[6][WORD_FEATURES.index("文節-区切")] = 1.0
    # 7: 儲かる
    word_features[7][WORD_FEATURES.index("基本句-主辞")] = 1.0
    # 8: 。
    word_features[8][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_features[8][WORD_FEATURES.index("文節-区切")] = 1.0

    base_phrase_features = [
        [0.0] * len(BASE_PHRASE_FEATURES) for _ in range(max_seq_length)
    ]
    # 0: 風
    base_phrase_features[0][BASE_PHRASE_FEATURES.index("体言")] = 1.0
    # 2: 吹く
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("用言:動")] = 1.0
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_features[2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    # 5: 桶屋
    base_phrase_features[5][BASE_PHRASE_FEATURES.index("体言")] = 1.0
    # 7: 儲かる
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("用言:動")] = 1.0
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_features[7][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    assert encoding["base_phrase_features"].tolist() == base_phrase_features

    dependencies = [[0] * max_seq_length for _ in range(max_seq_length)]
    # 0: 風 -> 2: 吹く
    dependencies[0][2] = 1
    # 1: が -> 0: 風
    dependencies[1][0] = 1
    # 2: 吹く -> ROOT (TODO)
    # 3: 。 -> 2: 吹く
    dependencies[3][2] = 1
    # 4: すると -> 7: 儲かる
    dependencies[4][7] = 1
    # 5: 桶屋 -> 7: 儲かる
    dependencies[5][7] = 1
    # 6: が -> 5: 桶屋
    dependencies[6][5] = 1
    # 7: 儲かる -> ROOT (TODO)
    # 8: 。 -> 7: 儲かる
    dependencies[8][7] = 1
    assert encoding["dependencies"].tolist() == dependencies
