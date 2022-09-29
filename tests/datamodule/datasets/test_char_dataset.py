import textwrap
from pathlib import Path

from rhoknp import Document

from kwja.datamodule.datasets.char_dataset import CharDataset
from kwja.utils.constants import IGNORE_INDEX, SEG_TYPES

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")

# TODO: use roberta
char_dataset_kwargs = dict(
    document_split_stride=1,
    model_name_or_path="ku-nlp/roberta-base-japanese-char-wwm",
    max_seq_length=512,
    tokenizer_kwargs={"do_word_tokenize": False},
)


def test_init():
    _ = CharDataset(str(path), **char_dataset_kwargs)


def test_getitem():
    max_seq_length: int = char_dataset_kwargs["max_seq_length"]
    dataset = CharDataset(str(path), **char_dataset_kwargs)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "seg_types" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["seg_types"].shape == (max_seq_length,)


def test_encode():
    max_seq_length = 20
    dataset = CharDataset(str(path), **(char_dataset_kwargs | {"max_seq_length": max_seq_length}))
    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:1 KNP:5.0-2ad4f6df
            * 2D <BGH:率/りつ><文頭><サ変><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><正規化代表表記:大学/だいがく+進学/しんがく+率/りつ><主辞代表表記:率/りつ><主辞’代表表記:進学/しんがく+率/りつ>
            + 1D <SM-主体><SM-場所><SM-組織><BGH:大学/だいがく><文節内><係:文節内><文頭><体言><名詞項候補><先行詞候補><正規化代表表記:大学/だいがく>
            大学 だいがく 大学 名詞 6 普通名詞 1 * 0 * 0 "代表表記:大学/だいがく ドメイン:教育・学習 カテゴリ:場所-施設 組織名末尾" <代表表記:大学/だいがく><ドメイン:教育・学習><カテゴリ:場所-施設><組織名末尾><正規化代表表記:大学/だいがく><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始>
            + 2D <BGH:進学/しんがく><文節内><係:文節内><サ変><体言><名詞項候補><先行詞候補><非用言格解析:動><照応ヒント:係><態:未定><正規化代表表記:進学/しんがく>
            進学 しんがく 進学 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:進学/しんがく ドメイン:教育・学習 カテゴリ:抽象物" <代表表記:進学/しんがく><ドメイン:教育・学習><カテゴリ:抽象物><正規化代表表記:進学/しんがく><漢字><かな漢字><名詞相当語><サ変><自立><複合←><内容語><タグ単位始>
            + 4D <BGH:率/りつ><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><一文字漢字><名詞項候補><先行詞候補><正規化代表表記:率/りつ><主辞代表表記:率/りつ><主辞’代表表記:進学/しんがく+率/りつ><Wikipedia上位語:割り合い/わりあい><Wikipediaエントリ:進学率><解析格:ガ>
            率 りつ 率 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0 "代表表記:率/りつ カテゴリ:数量 同義:副詞:割り合い/わりあい 内容語" <代表表記:率/りつ><カテゴリ:数量><同義:副詞:割り合い/わりあい><内容語><正規化代表表記:率/りつ><Wikipedia上位語:割り合い/わりあい:1-2><Wikipediaエントリ:進学率:1-2><漢字><かな漢字><名詞相当語><付属><タグ単位始><文節主辞>
            は は は 助詞 9 副助詞 2 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * 2D <カウンタ:％><数量><ニ><助詞><体言><係:ニ格><区切:0-0><格要素><連用要素><正規化代表表記:５４・４/５４・４+％/ぱーせんと><主辞代表表記:５４・４/５４・４+％/ぱーせんと>
            + 4D <カウンタ:％><数量><ニ><助詞><体言><係:ニ格><区切:0-0><格要素><連用要素><省略解析なし><正規化代表表記:５４・４/５４・４+％/ぱーせんと><主辞代表表記:５４・４/５４・４+％/ぱーせんと><解析格:ニ>
            ５４・４ ５４・４ ５４・４ 名詞 6 数詞 7 * 0 * 0 "カテゴリ:数量 未知語:数字 疑似代表表記 代表表記:５４・４/５４・４" <カテゴリ:数量><未知語:数字><疑似代表表記><代表表記:５４・４/５４・４><正規化代表表記:５４・４/５４・４><記英数カ><数字><名詞相当語><自立><内容語><タグ単位始><文節始>
            ％ ぱーせんと ％ 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:％/ぱーせんと 準内容語" <代表表記:％/ぱーせんと><準内容語><正規化代表表記:％/ぱーせんと><カウンタ><記英数カ><英記号><記号><付属><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            に に に 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
            * -1D <BGH:達する/たっする><文末><時制:過去><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:達する/たっする><主辞代表表記:達する/たっする>
            + -1D <BGH:達する/たっする><文末><時制:過去><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:達する/たっする><主辞代表表記:達する/たっする><用言代表表記:達する/たっする><節-区切><節-主辞><主題格:一人称優位><格関係2:ガ:率><格関係3:ニ:％><格解析結果:達する/たっする:動14:ガ/N/率/2/0/1;ニ/C/％/3/0/1;デ/U/-/-/-/-;時間/U/-/-/-/-><標準用言代表表記:達する/たっする>
            達した たっした 達する 動詞 2 * 0 サ変動詞 16 タ形 10 "代表表記:達する/たっする" <代表表記:達する/たっする><正規化代表表記:達する/たっする><かな漢字><活用語><表現文末><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            。 。 。 特殊 1 句点 1 * 0 * 0 NIL <英記号><記号><文末><付属>
            EOS
            """
        )
    )
    examples = dataset._load_examples([document])
    encoding = dataset.encode(examples[0])

    seg_types = [IGNORE_INDEX for _ in range(max_seq_length)]
    # 1: 大
    seg_types[1] = SEG_TYPES.index("B")
    # 2: 学
    seg_types[2] = SEG_TYPES.index("I")
    # 3: 進
    seg_types[3] = SEG_TYPES.index("B")
    # 4: 学
    seg_types[4] = SEG_TYPES.index("I")
    # 5: 率
    seg_types[5] = SEG_TYPES.index("B")
    # 6: は
    seg_types[6] = SEG_TYPES.index("B")
    # 7: ５
    seg_types[7] = SEG_TYPES.index("B")
    # 8: ４
    seg_types[8] = SEG_TYPES.index("I")
    # 9: ・
    seg_types[9] = SEG_TYPES.index("I")
    # 10: ４
    seg_types[10] = SEG_TYPES.index("I")
    # 11: ％
    seg_types[11] = SEG_TYPES.index("B")
    # 12: に
    seg_types[12] = SEG_TYPES.index("B")
    # 13: 達
    seg_types[13] = SEG_TYPES.index("B")
    # 14: し
    seg_types[14] = SEG_TYPES.index("I")
    # 15: た
    seg_types[15] = SEG_TYPES.index("I")
    # 16: 。
    seg_types[16] = SEG_TYPES.index("B")
    assert encoding["seg_types"].tolist() == seg_types
