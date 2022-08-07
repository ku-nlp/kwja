import textwrap
from pathlib import Path

from rhoknp import Document

from jula.datamodule.datasets.char_dataset import CharDataset
from jula.utils.constants import ENE_TYPE_BIES, IGNORE_INDEX

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")
wiki_ene_dic_path = here.joinpath("wiki_ene_dic")


def test_init():
    _ = CharDataset(path=str(path), wiki_ene_dic_path=str(wiki_ene_dic_path), max_seq_length=512)


def test_getitem():
    max_seq_length = 512
    max_ene_num = 3
    # TODO: use roberta
    dataset = CharDataset(
        path=str(path),
        wiki_ene_dic_path=str(wiki_ene_dic_path),
        max_ene_num=max_ene_num,
        model_name_or_path="cl-tohoku/bert-base-japanese-char",
        max_seq_length=max_seq_length,
        tokenizer_kwargs={"do_word_tokenize": False},
    )
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "ene_ids" in item
        assert "seg_labels" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["ene_ids"].shape == (max_ene_num, max_seq_length)
        assert item["seg_labels"].shape == (max_seq_length,)

        cls_token_position = item["input_ids"].tolist().index(dataset.tokenizer.cls_token_id)
        sep_token_position = item["input_ids"].tolist().index(dataset.tokenizer.sep_token_id)
        assert item["seg_labels"][cls_token_position].tolist() == IGNORE_INDEX
        assert item["seg_labels"][sep_token_position].tolist() == IGNORE_INDEX


def test_encode():
    max_seq_length = 20
    max_ene_num = 3
    dataset = CharDataset(
        path=str(path),
        wiki_ene_dic_path=str(wiki_ene_dic_path),
        max_ene_num=max_ene_num,
        model_name_or_path="cl-tohoku/bert-base-japanese-char",
        max_seq_length=max_seq_length,
        tokenizer_kwargs={"do_word_tokenize": False},
    )
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
    encoding = dataset.encode(document)

    ene_ids = [[ENE_TYPE_BIES.index("PAD")] * max_seq_length for _ in range(max_ene_num)]
    # 0: 大学
    ene_ids[0][1] = ENE_TYPE_BIES.index("B-0")
    ene_ids[0][2] = ENE_TYPE_BIES.index("E-0")
    # 1: 進学
    ene_ids[0][3] = ENE_TYPE_BIES.index("B-0")
    ene_ids[0][4] = ENE_TYPE_BIES.index("E-0")
    ene_ids[1][4] = ENE_TYPE_BIES.index("I-0")
    # 2: 率
    ene_ids[0][5] = ENE_TYPE_BIES.index("E-0")
    # 3: は
    # 4: ５４・４
    # 5: ％
    # 6: に
    # 7: 達した
    # 8: 。
    assert encoding["ene_ids"].tolist() == ene_ids
