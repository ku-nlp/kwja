from pathlib import Path
from textwrap import dedent

import jaconv
from rhoknp import Sentence
from transformers import PreTrainedTokenizerBase

from kwja.callbacks.seq2seq_module_writer import Seq2SeqModuleWriter
from kwja.utils.constants import NEW_LINE_TOKEN, NO_CANON_TOKEN, NO_READING_TOKEN
from kwja.utils.seq2seq_format import get_sent_from_seq2seq_format, get_seq2seq_format

input_seq2seq_formats = [
    dedent(
        f"""\
        計算 けいさん 計算 計算/けいさん
        機 き 機 機/き
        に に に {NO_CANON_TOKEN}
        よる よる よる 因る/よる
        言語 げんご 言語 言語/げんご
        理解 りかい 理解 理解/りかい
        を を を {NO_CANON_TOKEN}
        実現 じつげん 実現 実現/じつげん
        する する する する/する
        """
    ),
    dedent(
        """\
        また また また 又/また
        ， ， ， ，/，
        校区 こうく 校区 校区/こうく
        で で で で/で
        行わ おこなわ 行う 行う/おこなう
        れる れる れる れる/れる
        事業 じぎょう 事業 事業/じぎょう
        や や や や/や
        防犯 ぼうはん 防犯 防犯/ぼうはん
        など など など など/など
        校区 こうく 校区 校区/こうく
        の の の の/の
        情報 じょうほう 情報 情報/じょうほう
        も も も も/も
        記載 きさい 記載 記載/きさい
        さ さ する する/する
        れて れて れる れる/れる
        い い いる いる/いる
        ます ます ます ます/ます
        。 。 。 。/。
        """
    ),
    dedent(
        """\
        「 「 「 「/「
        核 かく 核 核/かく
        の の の の/の
        歴史 れきし 歴史 歴史/れきし
        … … … …/…
        ヒロシマ ひろしま ヒロシマ ヒロシマ/ひろしま
        、 、 、 、/、
        ナガサキ ながさき ナガサキ ナガサキ/ながさき
        を を を を/を
        超えて こえて 超える 超える/こえる
        」 」 」 」/」
        。 。 。 。/。
        """
    ),
    dedent(
        f"""\
        後 あと 後 後/あと
        一 ついたち 一 一/いち
        日 {NO_READING_TOKEN} 日 日/にち
        まで まで まで まで/まで
        ！ ！ ！ ！/！
        ？ ？ ？ ？/？
        """
    ),
]

output_seq2seq_formats = [
    dedent(
        f"""\
        計算 けいさん 計算 計算/けいさん
        機 き 機 機/き
        に に に {NO_CANON_TOKEN}
        よる よる よる 因る/よる
        言語 げんご 言語 言語/げんご
        理解 りかい 理解 理解/りかい
        を を を {NO_CANON_TOKEN}
        実現 じつげん 実現 実現/じつげん
        する する する する/する
        """
    ),
    dedent(
        """\
        また また また 又/また
        , , , ,/,
        校区 こうく 校区 校区/こうく
        で で で で/で
        行わ おこなわ 行う 行う/おこなう
        れる れる れる れる/れる
        事業 じぎょう 事業 事業/じぎょう
        や や や や/や
        防犯 ぼうはん 防犯 防犯/ぼうはん
        など など など など/など
        校区 こうく 校区 校区/こうく
        の の の の/の
        情報 じょうほう 情報 情報/じょうほう
        も も も も/も
        記載 きさい 記載 記載/きさい
        さ さ する する/する
        れて れて れる れる/れる
        い い いる いる/いる
        ます ます ます ます/ます
        。 。 。 。/。
        """
    ),
    dedent(
        """\
        「 「 「 「/「
        核 かく 核 核/かく
        の の の の/の
        歴史 れきし 歴史 歴史/れきし
        … … … …/…
        ヒロシマ ひろしま ヒロシマ ヒロシマ/ひろしま
        、 、 、 、/、
        ナガサキ ながさき ナガサキ ナガサキ/ながさき
        を を を を/を
        超えて こえて 超える 超える/こえる
        」 」 」 」/」
        。 。 。 。/。
        """
    ),
    dedent(
        f"""\
        後 あと 後 後/あと
        一 ついたち 一 一/いち
        日 {NO_READING_TOKEN} 日 日/にち
        まで まで まで まで/まで
        ! ! ! !/!
        ? ? ? ?/?
        """
    ),
]


def test_get_seq2seq_format(fixture_data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    juman_dir: Path = fixture_data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            assert get_seq2seq_format(sent) == input_seq2seq_formats[idx]


def test_get_sent_from_seq2seq_format(fixture_data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_module_writer = Seq2SeqModuleWriter(seq2seq_tokenizer)
    juman_dir: Path = fixture_data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            expected_sent = Sentence.from_jumanpp(f.read())

        shaped_output: str = seq2seq_module_writer.shape(
            output_seq2seq_formats[idx].replace("\n", NEW_LINE_TOKEN) + "</s>"
        )
        actual_sent = get_sent_from_seq2seq_format(shaped_output)
        assert len(actual_sent.morphemes) == len(expected_sent.morphemes)
        for actual_morpheme, expected_morpheme in zip(actual_sent.morphemes, expected_sent.morphemes):
            if expected_morpheme.reading == "\u3000":
                expected_reading: str = NO_READING_TOKEN
            elif "/" in expected_morpheme.reading:
                expected_reading = expected_morpheme.reading.split("/")[0]
            else:
                expected_reading = expected_morpheme.reading

            actual_canon: str = str(actual_morpheme.canon) if actual_morpheme.canon != f"{NO_CANON_TOKEN}" else "NIL"
            expected_canon: str = expected_morpheme.canon if expected_morpheme.canon is not None else "NIL"

            if expected_morpheme.pos == "特殊":
                assert jaconv.z2h(actual_morpheme.surf, ascii=True, digit=True) == jaconv.z2h(
                    expected_morpheme.surf, ascii=True, digit=True
                )
                assert jaconv.z2h(actual_morpheme.reading, ascii=True, digit=True) == jaconv.z2h(
                    expected_reading, ascii=True, digit=True
                )
                assert jaconv.z2h(actual_morpheme.lemma, ascii=True, digit=True) == jaconv.z2h(
                    expected_morpheme.lemma, ascii=True, digit=True
                )
                assert jaconv.z2h(actual_canon, ascii=True, digit=True) == jaconv.z2h(
                    expected_canon, ascii=True, digit=True
                )
            else:
                assert actual_morpheme.surf == expected_morpheme.surf
                assert actual_morpheme.reading == expected_reading
                assert actual_morpheme.lemma == expected_morpheme.lemma

            assert actual_morpheme.pos == "未定義語"
            assert actual_morpheme.pos_id == 15
            assert actual_morpheme.subpos == "その他"
            assert actual_morpheme.subpos_id == 1
            assert actual_morpheme.conjtype == "*"
            assert actual_morpheme.conjtype_id == 0
            assert actual_morpheme.conjform == "*"
            assert actual_morpheme.conjform_id == 0
