from pathlib import Path
from textwrap import dedent
from typing import List

import jaconv
from rhoknp import Sentence
from transformers import PreTrainedTokenizerBase

from kwja.utils.constants import (
    CANON_TOKEN,
    FULL_SPACE_TOKEN,
    HALF_SPACE_TOKEN1,
    LEMMA_TOKEN,
    NO_CANON_TOKEN,
    READING_TOKEN,
    SURF_TOKEN,
    TRIPLE_DOT_TOKEN,
)
from kwja.utils.seq2seq_format import Seq2SeqFormatter

texts: List[str] = [
    "計算機による言語理解を実現する",
    "また，校区で行われる事業や防犯など校区の情報も記載されています。",
    f"「核の歴史{TRIPLE_DOT_TOKEN}ヒロシマ、ナガサキを超えて」。",
    "後一日まで！？./",
    f"ＪＵＭＰ{FULL_SPACE_TOKEN}ＣＯＭＩＣＳ{HALF_SPACE_TOKEN1}",
]

seq2seq_formats: List[str] = [
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
        f"""\
        「 「 「 「/「
        核 かく 核 核/かく
        の の の の/の
        歴史 れきし 歴史 歴史/れきし
        {TRIPLE_DOT_TOKEN} {TRIPLE_DOT_TOKEN} {TRIPLE_DOT_TOKEN} {TRIPLE_DOT_TOKEN}/{TRIPLE_DOT_TOKEN}
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
        日 {FULL_SPACE_TOKEN} 日 日/にち
        まで まで まで まで/まで
        ！ ！ ！ ！/！
        ？ ？ ？ ？/？
        . . . ./.
        / / / ///
        """
    ),
    dedent(
        f"""\
        ＪＵＭＰ ＪＵＭＰ ＪＵＭＰ ＪＵＭＰ/ＪＵＭＰ
        {FULL_SPACE_TOKEN} {FULL_SPACE_TOKEN} {FULL_SPACE_TOKEN} /
        ＣＯＭＩＣＳ ＣＯＭＩＣＳ ＣＯＭＩＣＳ ＣＯＭＩＣＳ/ＣＯＭＩＣＳ
        {HALF_SPACE_TOKEN1} {HALF_SPACE_TOKEN1} {HALF_SPACE_TOKEN1} /
        """
    ),
]


tokenizeds: List[List[str]] = [
    [
        f"{SURF_TOKEN}",
        "計算",
        f"{READING_TOKEN}",
        "けい",
        "さん",
        f"{LEMMA_TOKEN}",
        "計算",
        f"{CANON_TOKEN}",
        "計算",
        "/",
        "けい",
        "さん",
        f"{SURF_TOKEN}",
        "機",
        f"{READING_TOKEN}",
        "き",
        f"{LEMMA_TOKEN}",
        "機",
        f"{CANON_TOKEN}",
        "機",
        "/",
        "き",
        f"{SURF_TOKEN}",
        "に",
        f"{READING_TOKEN}",
        "に",
        f"{LEMMA_TOKEN}",
        "に",
        f"{CANON_TOKEN}",
        "<extra_id_4>",
        f"{SURF_TOKEN}",
        "よ",
        "る",
        f"{READING_TOKEN}",
        "よ",
        "る",
        f"{LEMMA_TOKEN}",
        "よ",
        "る",
        f"{CANON_TOKEN}",
        "因",
        "る",
        "/",
        "よ",
        "る",
        f"{SURF_TOKEN}",
        "言語",
        f"{READING_TOKEN}",
        "げん",
        "ご",
        f"{LEMMA_TOKEN}",
        "言語",
        f"{CANON_TOKEN}",
        "言語",
        "/",
        "げん",
        "ご",
        f"{SURF_TOKEN}",
        "理解",
        f"{READING_TOKEN}",
        "り",
        "かい",
        f"{LEMMA_TOKEN}",
        "理解",
        f"{CANON_TOKEN}",
        "理解",
        "/",
        "り",
        "かい",
        f"{SURF_TOKEN}",
        "を",
        f"{READING_TOKEN}",
        "を",
        f"{LEMMA_TOKEN}",
        "を",
        f"{CANON_TOKEN}",
        "<extra_id_4>",
        f"{SURF_TOKEN}",
        "実現",
        f"{READING_TOKEN}",
        "じ",
        "つ",
        "げん",
        f"{LEMMA_TOKEN}",
        "実現",
        f"{CANON_TOKEN}",
        "実現",
        "/",
        "じ",
        "つ",
        "げん",
        f"{SURF_TOKEN}",
        "する",
        f"{READING_TOKEN}",
        "する",
        f"{LEMMA_TOKEN}",
        "する",
        f"{CANON_TOKEN}",
        "する",
        "/",
        "する",
    ],
    [
        f"{SURF_TOKEN}",
        "▁また",
        f"{READING_TOKEN}",
        "▁また",
        f"{LEMMA_TOKEN}",
        "▁また",
        f"{CANON_TOKEN}",
        "又",
        "/",
        "また",
        f"{SURF_TOKEN}",
        ",",
        f"{READING_TOKEN}",
        ",",
        f"{LEMMA_TOKEN}",
        ",",
        f"{CANON_TOKEN}",
        ",",
        "/",
        ",",
        f"{SURF_TOKEN}",
        "校",
        "区",
        f"{READING_TOKEN}",
        "こう",
        "く",
        f"{LEMMA_TOKEN}",
        "校",
        "区",
        f"{CANON_TOKEN}",
        "校",
        "区",
        "/",
        "こう",
        "く",
        f"{SURF_TOKEN}",
        "で",
        f"{READING_TOKEN}",
        "で",
        f"{LEMMA_TOKEN}",
        "で",
        f"{CANON_TOKEN}",
        "で",
        "/",
        "で",
        f"{SURF_TOKEN}",
        "行",
        "わ",
        f"{READING_TOKEN}",
        "おこ",
        "な",
        "わ",
        f"{LEMMA_TOKEN}",
        "行う",
        f"{CANON_TOKEN}",
        "行う",
        "/",
        "おこ",
        "な",
        "う",
        f"{SURF_TOKEN}",
        "れる",
        f"{READING_TOKEN}",
        "れる",
        f"{LEMMA_TOKEN}",
        "れる",
        f"{CANON_TOKEN}",
        "れる",
        "/",
        "れる",
        f"{SURF_TOKEN}",
        "事業",
        f"{READING_TOKEN}",
        "じ",
        "ぎ",
        "ょう",
        f"{LEMMA_TOKEN}",
        "事業",
        f"{CANON_TOKEN}",
        "事業",
        "/",
        "じ",
        "ぎ",
        "ょう",
        f"{SURF_TOKEN}",
        "や",
        f"{READING_TOKEN}",
        "や",
        f"{LEMMA_TOKEN}",
        "や",
        f"{CANON_TOKEN}",
        "や",
        "/",
        "や",
        f"{SURF_TOKEN}",
        "防犯",
        f"{READING_TOKEN}",
        "ぼう",
        "は",
        "ん",
        f"{LEMMA_TOKEN}",
        "防犯",
        f"{CANON_TOKEN}",
        "防犯",
        "/",
        "ぼう",
        "は",
        "ん",
        f"{SURF_TOKEN}",
        "など",
        f"{READING_TOKEN}",
        "など",
        f"{LEMMA_TOKEN}",
        "など",
        f"{CANON_TOKEN}",
        "など",
        "/",
        "など",
        f"{SURF_TOKEN}",
        "校",
        "区",
        f"{READING_TOKEN}",
        "こう",
        "く",
        f"{LEMMA_TOKEN}",
        "校",
        "区",
        f"{CANON_TOKEN}",
        "校",
        "区",
        "/",
        "こう",
        "く",
        f"{SURF_TOKEN}",
        "の",
        f"{READING_TOKEN}",
        "の",
        f"{LEMMA_TOKEN}",
        "の",
        f"{CANON_TOKEN}",
        "の",
        "/",
        "の",
        f"{SURF_TOKEN}",
        "情報",
        f"{READING_TOKEN}",
        "じ",
        "ょう",
        "ほう",
        f"{LEMMA_TOKEN}",
        "情報",
        f"{CANON_TOKEN}",
        "情報",
        "/",
        "じ",
        "ょう",
        "ほう",
        f"{SURF_TOKEN}",
        "も",
        f"{READING_TOKEN}",
        "も",
        f"{LEMMA_TOKEN}",
        "も",
        f"{CANON_TOKEN}",
        "も",
        "/",
        "も",
        f"{SURF_TOKEN}",
        "記載",
        f"{READING_TOKEN}",
        "き",
        "さい",
        f"{LEMMA_TOKEN}",
        "記載",
        f"{CANON_TOKEN}",
        "記載",
        "/",
        "き",
        "さい",
        f"{SURF_TOKEN}",
        "さ",
        f"{READING_TOKEN}",
        "さ",
        f"{LEMMA_TOKEN}",
        "する",
        f"{CANON_TOKEN}",
        "する",
        "/",
        "する",
        f"{SURF_TOKEN}",
        "れて",
        f"{READING_TOKEN}",
        "れて",
        f"{LEMMA_TOKEN}",
        "れる",
        f"{CANON_TOKEN}",
        "れる",
        "/",
        "れる",
        f"{SURF_TOKEN}",
        "い",
        f"{READING_TOKEN}",
        "い",
        f"{LEMMA_TOKEN}",
        "いる",
        f"{CANON_TOKEN}",
        "いる",
        "/",
        "いる",
        f"{SURF_TOKEN}",
        "ます",
        f"{READING_TOKEN}",
        "ます",
        f"{LEMMA_TOKEN}",
        "ます",
        f"{CANON_TOKEN}",
        "ます",
        "/",
        "ます",
        f"{SURF_TOKEN}",
        "。",
        f"{READING_TOKEN}",
        "。",
        f"{LEMMA_TOKEN}",
        "。",
        f"{CANON_TOKEN}",
        "。",
        "/",
        "。",
    ],
    [
        f"{SURF_TOKEN}",
        "▁「",
        f"{READING_TOKEN}",
        "▁「",
        f"{LEMMA_TOKEN}",
        "▁「",
        f"{CANON_TOKEN}",
        "▁「",
        "/",
        "「",
        f"{SURF_TOKEN}",
        "核",
        f"{READING_TOKEN}",
        "かく",
        f"{LEMMA_TOKEN}",
        "核",
        f"{CANON_TOKEN}",
        "核",
        "/",
        "かく",
        f"{SURF_TOKEN}",
        "の",
        f"{READING_TOKEN}",
        "の",
        f"{LEMMA_TOKEN}",
        "の",
        f"{CANON_TOKEN}",
        "の",
        "/",
        "の",
        f"{SURF_TOKEN}",
        "歴史",
        f"{READING_TOKEN}",
        "れ",
        "き",
        "し",
        f"{LEMMA_TOKEN}",
        "歴史",
        f"{CANON_TOKEN}",
        "歴史",
        "/",
        "れ",
        "き",
        "し",
        f"{SURF_TOKEN}",
        "<extra_id_7>",
        f"{READING_TOKEN}",
        "<extra_id_7>",
        f"{LEMMA_TOKEN}",
        "<extra_id_7>",
        f"{CANON_TOKEN}",
        "<extra_id_7>",
        "▁",
        "/",
        "<extra_id_7>",
        f"{SURF_TOKEN}",
        "ヒロ",
        "シマ",
        f"{READING_TOKEN}",
        "ひろ",
        "しま",
        f"{LEMMA_TOKEN}",
        "ヒロ",
        "シマ",
        f"{CANON_TOKEN}",
        "ヒロ",
        "シマ",
        "/",
        "ひろ",
        "しま",
        f"{SURF_TOKEN}",
        "、",
        f"{READING_TOKEN}",
        "、",
        f"{LEMMA_TOKEN}",
        "、",
        f"{CANON_TOKEN}",
        "、",
        "/",
        "、",
        f"{SURF_TOKEN}",
        "ナ",
        "ガ",
        "サ",
        "キ",
        f"{READING_TOKEN}",
        "な",
        "が",
        "さ",
        "き",
        f"{LEMMA_TOKEN}",
        "ナ",
        "ガ",
        "サ",
        "キ",
        f"{CANON_TOKEN}",
        "ナ",
        "ガ",
        "サ",
        "キ",
        "/",
        "な",
        "が",
        "さ",
        "き",
        f"{SURF_TOKEN}",
        "を",
        f"{READING_TOKEN}",
        "を",
        f"{LEMMA_TOKEN}",
        "を",
        f"{CANON_TOKEN}",
        "を",
        "/",
        "を",
        f"{SURF_TOKEN}",
        "超",
        "えて",
        f"{READING_TOKEN}",
        "こ",
        "えて",
        f"{LEMMA_TOKEN}",
        "超",
        "える",
        f"{CANON_TOKEN}",
        "超",
        "える",
        "/",
        "こ",
        "える",
        f"{SURF_TOKEN}",
        "」",
        f"{READING_TOKEN}",
        "」",
        f"{LEMMA_TOKEN}",
        "」",
        f"{CANON_TOKEN}",
        "」",
        "/",
        "」",
        f"{SURF_TOKEN}",
        "。",
        f"{READING_TOKEN}",
        "。",
        f"{LEMMA_TOKEN}",
        "。",
        f"{CANON_TOKEN}",
        "。",
        "/",
        "。",
    ],
    [
        f"{SURF_TOKEN}",
        "後",
        f"{READING_TOKEN}",
        "あと",
        f"{LEMMA_TOKEN}",
        "後",
        f"{CANON_TOKEN}",
        "後",
        "/",
        "あと",
        f"{SURF_TOKEN}",
        "一",
        f"{READING_TOKEN}",
        "ついた",
        "ち",
        f"{LEMMA_TOKEN}",
        "一",
        f"{CANON_TOKEN}",
        "一",
        "/",
        "いち",
        f"{SURF_TOKEN}",
        "日",
        f"{READING_TOKEN}",
        "<extra_id_5>",
        f"{LEMMA_TOKEN}",
        "日",
        f"{CANON_TOKEN}",
        "日",
        "/",
        "にち",
        f"{SURF_TOKEN}",
        "まで",
        f"{READING_TOKEN}",
        "まで",
        f"{LEMMA_TOKEN}",
        "まで",
        f"{CANON_TOKEN}",
        "まで",
        "/",
        "まで",
        f"{SURF_TOKEN}",
        "!",
        f"{READING_TOKEN}",
        "!",
        f"{LEMMA_TOKEN}",
        "!",
        f"{CANON_TOKEN}",
        "!",
        "/",
        "!",
        f"{SURF_TOKEN}",
        "?",
        f"{READING_TOKEN}",
        "?",
        f"{LEMMA_TOKEN}",
        "?",
        f"{CANON_TOKEN}",
        "?",
        "/?",
        f"{SURF_TOKEN}",
        ".",
        f"{READING_TOKEN}",
        ".",
        f"{LEMMA_TOKEN}",
        ".",
        f"{CANON_TOKEN}",
        ".",
        "/",
        ".",
        f"{SURF_TOKEN}",
        "/",
        f"{READING_TOKEN}",
        "/",
        f"{LEMMA_TOKEN}",
        "/",
        f"{CANON_TOKEN}",
        "▁///",
    ],
    [
        f"{SURF_TOKEN}",
        "▁J",
        "UMP",
        f"{READING_TOKEN}",
        "▁J",
        "UMP",
        f"{LEMMA_TOKEN}",
        "▁J",
        "UMP",
        f"{CANON_TOKEN}",
        "▁J",
        "UMP",
        "/",
        "J",
        "UMP",
        f"{SURF_TOKEN}",
        "<extra_id_5>",
        f"{READING_TOKEN}",
        "<extra_id_5>",
        f"{LEMMA_TOKEN}",
        "<extra_id_5>",
        f"{CANON_TOKEN}",
        "/",
        f"{SURF_TOKEN}",
        "COM",
        "ICS",
        f"{READING_TOKEN}",
        "COM",
        "ICS",
        f"{LEMMA_TOKEN}",
        "COM",
        "ICS",
        f"{CANON_TOKEN}",
        "COM",
        "ICS",
        "/",
        "COM",
        "ICS",
        f"{SURF_TOKEN}",
        "<extra_id_6>",
        f"{READING_TOKEN}",
        "<extra_id_6>",
        f"{LEMMA_TOKEN}",
        "<extra_id_6>",
        f"{CANON_TOKEN}",
        "/",
    ],
]


def test_sent_to_text(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            assert seq2seq_formatter.sent_to_text(sent) == texts[idx]


def test_get_seq2seq_format(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            seq2seq_format: List[str] = []
            for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
                splitted: List[str] = line.split(" ")
                seq2seq_format.extend(
                    f"{SURF_TOKEN} {splitted[0]} {READING_TOKEN} {splitted[1]} {LEMMA_TOKEN} {splitted[2]} {CANON_TOKEN} {splitted[3]}".split(
                        " "
                    )
                )
            assert seq2seq_formatter.sent_to_format(sent) == seq2seq_format


def test_tokenize(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            seq2seq_format: List[str] = seq2seq_formatter.sent_to_format(sent)
            tgt_tokens: List[str] = seq2seq_formatter.tokenize(seq2seq_format)
            assert tgt_tokens == tokenizeds[idx]


def test_get_sent_from_seq2seq_format(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            expected_sent = Sentence.from_jumanpp(f.read())

        seq2seq_format: List[str] = []
        for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
            splitted: List[str] = line.split(" ")
            seq2seq_format.extend(
                f"{SURF_TOKEN} {splitted[0]} {READING_TOKEN} {splitted[1]} {LEMMA_TOKEN} {splitted[2]} {CANON_TOKEN} {splitted[3]}".split(
                    " "
                )
            )
        actual_sent: Sentence = seq2seq_formatter.format_to_sent("".join(seq2seq_format))
        assert len(actual_sent.morphemes) == len(expected_sent.morphemes)
        for actual_morpheme, expected_morpheme in zip(actual_sent.morphemes, expected_sent.morphemes):
            if "/" in expected_morpheme.reading and len(expected_morpheme.reading) > 1:
                expected_reading: str = expected_morpheme.reading.split("/")[0]
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
