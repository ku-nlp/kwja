import textwrap
from pathlib import Path

from rhoknp import Document
from transformers import AutoTokenizer

from kwja.datamodule.datasets.char_dataset import CharDataset
from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")

tokenizer = AutoTokenizer.from_pretrained("ku-nlp/roberta-base-japanese-char-wwm", do_word_tokenize=False)


def test_init():
    _ = CharDataset(str(path), tokenizer, 1, max_seq_length=256)


def test_getitem():
    max_seq_length: int = 256
    dataset = CharDataset(str(path), tokenizer, 1, max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "word_segmentation_labels" in item
        assert "word_norm_op_labels" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["word_segmentation_labels"].shape == (max_seq_length,)
        assert item["word_norm_op_labels"].shape == (max_seq_length,)


def test_encode():
    max_seq_length = 512
    dataset = CharDataset(str(path), tokenizer, 1, max_seq_length=max_seq_length)
    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:1 KNP:5.0-25425d33
            * 2D <BGH:率/りつ><文頭><サ変><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><正規化代表表記:進学/しんがく+率/りつ><主辞代表表記:率/りつ><主辞’代表表記:進学/しんがく+率/りつ>
            + 1D <BGH:進学/しんがく><文節内><係:文節内><文頭><サ変><体言><名詞項候補><先行詞候補><非用言格解析:動><照応ヒント:係><態:未定><正規化代表表記:進学/しんがく>
            進学 しんがく 進学 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:進学/しんがく ドメイン:教育・学習 カテゴリ:抽象物" <代表表記:進学/しんがく><ドメイン:教育・学習><カテゴリ:抽象物><正規化代表表記:進学/しんがく><漢字><かな漢字><名詞相当語><文頭><サ変><自立><内容語><タグ単位始><文節始>
            + 3D <BGH:率/りつ><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><一文字漢字><名詞項候補><先行詞候補><正規化代表表記:率/りつ><主辞代表表記:率/りつ><主辞’代表表記:進学/しんがく+率/りつ><Wikipedia上位語:割り合い/わりあい><Wikipediaエントリ:進学率><解析格:ガ>
            率 りつ 率 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0 "代表表記:率/りつ カテゴリ:数量 同義:副詞:割り合い/わりあい 内容語" <代表表記:率/りつ><カテゴリ:数量><同義:副詞:割り合い/わりあい><内容語><正規化代表表記:率/りつ><Wikipedia上位語:割り合い/わりあい:0-1><Wikipediaエントリ:進学率:0-1><漢字><かな漢字><名詞相当語><付属><タグ単位始><文節主辞>
            は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は" <代表表記:は/は><正規化代表表記:は/は><かな漢字><ひらがな><付属>
            * 2D <カウンタ:％><数量><ニ><助詞><体言><係:ニ格><区切:0-0><格要素><連用要素><正規化代表表記:５０/５０+％/ぱーせんと><主辞代表表記:５０/５０+％/ぱーせんと>
            + 3D <カウンタ:％><数量><ニ><助詞><体言><係:ニ格><区切:0-0><格要素><連用要素><省略解析なし><正規化代表表記:５０/５０+％/ぱーせんと><主辞代表表記:５０/５０+％/ぱーせんと><解析格:ニ>
            ５０ ５０ ５０ 名詞 6 数詞 7 * 0 * 0 "カテゴリ:数量 未知語:数字 疑似代表表記 代表表記:５０/５０" <カテゴリ:数量><未知語:数字><疑似代表表記><代表表記:５０/５０><正規化代表表記:５０/５０><記英数カ><数字><名詞相当語><自立><内容語><タグ単位始><文節始>
            ％ ぱーせんと ％ 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:％/ぱーせんと 準内容語" <代表表記:％/ぱーせんと><準内容語><正規化代表表記:％/ぱーせんと><カウンタ><記英数カ><英記号><記号><付属><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
            に に に 助詞 9 格助詞 1 * 0 * 0 "代表表記:に/に" <代表表記:に/に><正規化代表表記:に/に><かな漢字><ひらがな><付属>
            * -1D <BGH:達する/たっする><文末><〜たい><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><状態述語><モダリティ-意志><正規化代表表記:達する/たっする><主辞代表表記:達する/たっする>
            + -1D <BGH:達する/たっする><文末><〜たい><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><状態述語><モダリティ-意志><正規化代表表記:達する/たっする><主辞代表表記:達する/たっする><用言代表表記:達する/たっする+たい/たい><節-区切><節-主辞><主題格:一人称優位><格関係1:ガ:率><格関係2:ニ:％><格解析結果:達する/たっする+たい/たい:動1:ガ/N/率/1/0/1;ニ/C/％/2/0/1><標準用言代表表記:達する/たっする+たい/たい>
            達し たっし 達する 動詞 2 * 0 サ変動詞 16 基本連用形 8 "代表表記:達する/たっする" <代表表記:達する/たっする><正規化代表表記:達する/たっする><かな漢字><活用語><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭>
            たぁ た たい 接尾辞 14 形容詞性述語接尾辞 5 イ形容詞アウオ段 18 語幹 1 "代表表記:たい/たい 非標準表記:DSL" <代表表記:たい/たい><非標準表記:DSL><正規化代表表記:たい/たい><かな漢字><ひらがな><活用語><表現文末><付属><用言表記末尾><用言意味表記末尾>
            。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。" <代表表記:。/。><正規化代表表記:。/。><英記号><記号><文末><付属>
            EOS
            """
        )
    )
    examples = dataset._load_examples([document])
    encoding = dataset.encode(examples[0])

    word_segmentation_labels = [IGNORE_INDEX for _ in range(max_seq_length)]
    word_segmentation_labels[1] = WORD_SEGMENTATION_TAGS.index("B")  # 進
    word_segmentation_labels[2] = WORD_SEGMENTATION_TAGS.index("I")  # 学
    word_segmentation_labels[3] = WORD_SEGMENTATION_TAGS.index("B")  # 率
    word_segmentation_labels[4] = WORD_SEGMENTATION_TAGS.index("B")  # は
    word_segmentation_labels[5] = WORD_SEGMENTATION_TAGS.index("B")  # ５
    word_segmentation_labels[6] = WORD_SEGMENTATION_TAGS.index("I")  # ０
    word_segmentation_labels[7] = WORD_SEGMENTATION_TAGS.index("B")  # ％
    word_segmentation_labels[8] = WORD_SEGMENTATION_TAGS.index("B")  # に
    word_segmentation_labels[9] = WORD_SEGMENTATION_TAGS.index("B")  # 達
    word_segmentation_labels[10] = WORD_SEGMENTATION_TAGS.index("I")  # し
    word_segmentation_labels[11] = WORD_SEGMENTATION_TAGS.index("B")  # た
    word_segmentation_labels[12] = WORD_SEGMENTATION_TAGS.index("I")  # ぁ
    word_segmentation_labels[13] = WORD_SEGMENTATION_TAGS.index("B")  # 。
    assert encoding["word_segmentation_labels"].tolist() == word_segmentation_labels

    word_norm_op_labels = [IGNORE_INDEX for _ in range(max_seq_length)]
    word_norm_op_labels[1] = WORD_NORM_OP_TAGS.index("K")  # 進
    word_norm_op_labels[2] = WORD_NORM_OP_TAGS.index("K")  # 学
    word_norm_op_labels[3] = WORD_NORM_OP_TAGS.index("K")  # 率
    word_norm_op_labels[4] = WORD_NORM_OP_TAGS.index("K")  # は
    word_norm_op_labels[5] = WORD_NORM_OP_TAGS.index("K")  # ５
    word_norm_op_labels[6] = WORD_NORM_OP_TAGS.index("K")  # ０
    word_norm_op_labels[7] = WORD_NORM_OP_TAGS.index("K")  # ％
    word_norm_op_labels[8] = WORD_NORM_OP_TAGS.index("K")  # に
    word_norm_op_labels[9] = WORD_NORM_OP_TAGS.index("K")  # 達
    word_norm_op_labels[10] = WORD_NORM_OP_TAGS.index("K")  # し
    word_norm_op_labels[11] = WORD_NORM_OP_TAGS.index("K")  # た
    word_norm_op_labels[12] = WORD_NORM_OP_TAGS.index("D")  # ぁ
    word_norm_op_labels[13] = WORD_NORM_OP_TAGS.index("K")  # 。
    assert encoding["word_norm_op_labels"].tolist() == word_norm_op_labels
