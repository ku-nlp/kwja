from pathlib import Path
from textwrap import dedent
from typing import Dict, List

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

tokenizeds: List[List[str]] = [
    [
        SURF_TOKEN,
        "計算",
        READING_TOKEN,
        "けい",
        "さん",
        LEMMA_TOKEN,
        "計算",
        CANON_TOKEN,
        "計算",
        "/",
        "けい",
        "さん",
        SURF_TOKEN,
        "機",
        READING_TOKEN,
        "き",
        LEMMA_TOKEN,
        "機",
        CANON_TOKEN,
        "機",
        "/",
        "き",
        SURF_TOKEN,
        "に",
        READING_TOKEN,
        "に",
        LEMMA_TOKEN,
        "に",
        CANON_TOKEN,
        NO_CANON_TOKEN,
        SURF_TOKEN,
        "よ",
        "る",
        READING_TOKEN,
        "よ",
        "る",
        LEMMA_TOKEN,
        "よ",
        "る",
        CANON_TOKEN,
        "因",
        "る",
        "/",
        "よ",
        "る",
        SURF_TOKEN,
        "言語",
        READING_TOKEN,
        "げん",
        "ご",
        LEMMA_TOKEN,
        "言語",
        CANON_TOKEN,
        "言語",
        "/",
        "げん",
        "ご",
        SURF_TOKEN,
        "理解",
        READING_TOKEN,
        "り",
        "かい",
        LEMMA_TOKEN,
        "理解",
        CANON_TOKEN,
        "理解",
        "/",
        "り",
        "かい",
        SURF_TOKEN,
        "を",
        READING_TOKEN,
        "を",
        LEMMA_TOKEN,
        "を",
        CANON_TOKEN,
        NO_CANON_TOKEN,
        SURF_TOKEN,
        "実現",
        READING_TOKEN,
        "じ",
        "つ",
        "げん",
        LEMMA_TOKEN,
        "実現",
        CANON_TOKEN,
        "実現",
        "/",
        "じ",
        "つ",
        "げん",
        SURF_TOKEN,
        "する",
        READING_TOKEN,
        "する",
        LEMMA_TOKEN,
        "する",
        CANON_TOKEN,
        "する",
        "/",
        "する",
        "</s>",
    ],
    [
        SURF_TOKEN,
        "▁また",
        READING_TOKEN,
        "▁また",
        LEMMA_TOKEN,
        "▁また",
        CANON_TOKEN,
        "又",
        "/",
        "また",
        SURF_TOKEN,
        ",",
        READING_TOKEN,
        ",",
        LEMMA_TOKEN,
        ",",
        CANON_TOKEN,
        ",",
        "/",
        ",",
        SURF_TOKEN,
        "校",
        "区",
        READING_TOKEN,
        "こう",
        "く",
        LEMMA_TOKEN,
        "校",
        "区",
        CANON_TOKEN,
        "校",
        "区",
        "/",
        "こう",
        "く",
        SURF_TOKEN,
        "で",
        READING_TOKEN,
        "で",
        LEMMA_TOKEN,
        "で",
        CANON_TOKEN,
        "で",
        "/",
        "で",
        SURF_TOKEN,
        "行",
        "わ",
        READING_TOKEN,
        "おこ",
        "な",
        "わ",
        LEMMA_TOKEN,
        "行う",
        CANON_TOKEN,
        "行う",
        "/",
        "おこ",
        "な",
        "う",
        SURF_TOKEN,
        "れる",
        READING_TOKEN,
        "れる",
        LEMMA_TOKEN,
        "れる",
        CANON_TOKEN,
        "れる",
        "/",
        "れる",
        SURF_TOKEN,
        "事業",
        READING_TOKEN,
        "じ",
        "ぎ",
        "ょう",
        LEMMA_TOKEN,
        "事業",
        CANON_TOKEN,
        "事業",
        "/",
        "じ",
        "ぎ",
        "ょう",
        SURF_TOKEN,
        "や",
        READING_TOKEN,
        "や",
        LEMMA_TOKEN,
        "や",
        CANON_TOKEN,
        "や",
        "/",
        "や",
        SURF_TOKEN,
        "防犯",
        READING_TOKEN,
        "ぼう",
        "は",
        "ん",
        LEMMA_TOKEN,
        "防犯",
        CANON_TOKEN,
        "防犯",
        "/",
        "ぼう",
        "は",
        "ん",
        SURF_TOKEN,
        "など",
        READING_TOKEN,
        "など",
        LEMMA_TOKEN,
        "など",
        CANON_TOKEN,
        "など",
        "/",
        "など",
        SURF_TOKEN,
        "校",
        "区",
        READING_TOKEN,
        "こう",
        "く",
        LEMMA_TOKEN,
        "校",
        "区",
        CANON_TOKEN,
        "校",
        "区",
        "/",
        "こう",
        "く",
        SURF_TOKEN,
        "の",
        READING_TOKEN,
        "の",
        LEMMA_TOKEN,
        "の",
        CANON_TOKEN,
        "の",
        "/",
        "の",
        SURF_TOKEN,
        "情報",
        READING_TOKEN,
        "じ",
        "ょう",
        "ほう",
        LEMMA_TOKEN,
        "情報",
        CANON_TOKEN,
        "情報",
        "/",
        "じ",
        "ょう",
        "ほう",
        SURF_TOKEN,
        "も",
        READING_TOKEN,
        "も",
        LEMMA_TOKEN,
        "も",
        CANON_TOKEN,
        "も",
        "/",
        "も",
        SURF_TOKEN,
        "記載",
        READING_TOKEN,
        "き",
        "さい",
        LEMMA_TOKEN,
        "記載",
        CANON_TOKEN,
        "記載",
        "/",
        "き",
        "さい",
        SURF_TOKEN,
        "さ",
        READING_TOKEN,
        "さ",
        LEMMA_TOKEN,
        "する",
        CANON_TOKEN,
        "する",
        "/",
        "する",
        SURF_TOKEN,
        "れて",
        READING_TOKEN,
        "れて",
        LEMMA_TOKEN,
        "れる",
        CANON_TOKEN,
        "れる",
        "/",
        "れる",
        SURF_TOKEN,
        "い",
        READING_TOKEN,
        "い",
        LEMMA_TOKEN,
        "いる",
        CANON_TOKEN,
        "いる",
        "/",
        "いる",
        SURF_TOKEN,
        "ます",
        READING_TOKEN,
        "ます",
        LEMMA_TOKEN,
        "ます",
        CANON_TOKEN,
        "ます",
        "/",
        "ます",
        SURF_TOKEN,
        "。",
        READING_TOKEN,
        "。",
        LEMMA_TOKEN,
        "。",
        CANON_TOKEN,
        "。",
        "/",
        "。",
        "</s>",
    ],
    [
        SURF_TOKEN,
        "▁「",
        READING_TOKEN,
        "▁「",
        LEMMA_TOKEN,
        "▁「",
        CANON_TOKEN,
        "▁「",
        "/",
        "「",
        SURF_TOKEN,
        "核",
        READING_TOKEN,
        "かく",
        LEMMA_TOKEN,
        "核",
        CANON_TOKEN,
        "核",
        "/",
        "かく",
        SURF_TOKEN,
        "の",
        READING_TOKEN,
        "の",
        LEMMA_TOKEN,
        "の",
        CANON_TOKEN,
        "の",
        "/",
        "の",
        SURF_TOKEN,
        "歴史",
        READING_TOKEN,
        "れ",
        "き",
        "し",
        LEMMA_TOKEN,
        "歴史",
        CANON_TOKEN,
        "歴史",
        "/",
        "れ",
        "き",
        "し",
        SURF_TOKEN,
        TRIPLE_DOT_TOKEN,
        READING_TOKEN,
        TRIPLE_DOT_TOKEN,
        LEMMA_TOKEN,
        TRIPLE_DOT_TOKEN,
        CANON_TOKEN,
        TRIPLE_DOT_TOKEN,
        "/",
        TRIPLE_DOT_TOKEN,
        SURF_TOKEN,
        "ヒロ",
        "シマ",
        READING_TOKEN,
        "ひろ",
        "しま",
        LEMMA_TOKEN,
        "ヒロ",
        "シマ",
        CANON_TOKEN,
        "ヒロ",
        "シマ",
        "/",
        "ひろ",
        "しま",
        SURF_TOKEN,
        "、",
        READING_TOKEN,
        "、",
        LEMMA_TOKEN,
        "、",
        CANON_TOKEN,
        "、",
        "/",
        "、",
        SURF_TOKEN,
        "ナ",
        "ガ",
        "サ",
        "キ",
        READING_TOKEN,
        "な",
        "が",
        "さ",
        "き",
        LEMMA_TOKEN,
        "ナ",
        "ガ",
        "サ",
        "キ",
        CANON_TOKEN,
        "ナ",
        "ガ",
        "サ",
        "キ",
        "/",
        "な",
        "が",
        "さ",
        "き",
        SURF_TOKEN,
        "を",
        READING_TOKEN,
        "を",
        LEMMA_TOKEN,
        "を",
        CANON_TOKEN,
        "を",
        "/",
        "を",
        SURF_TOKEN,
        "超",
        "えて",
        READING_TOKEN,
        "こ",
        "えて",
        LEMMA_TOKEN,
        "超",
        "える",
        CANON_TOKEN,
        "超",
        "える",
        "/",
        "こ",
        "える",
        SURF_TOKEN,
        "」",
        READING_TOKEN,
        "」",
        LEMMA_TOKEN,
        "」",
        CANON_TOKEN,
        "」",
        "/",
        "」",
        SURF_TOKEN,
        "。",
        READING_TOKEN,
        "。",
        LEMMA_TOKEN,
        "。",
        CANON_TOKEN,
        "。",
        "/",
        "。",
        "</s>",
    ],
    [
        SURF_TOKEN,
        "後",
        READING_TOKEN,
        "あと",
        LEMMA_TOKEN,
        "後",
        CANON_TOKEN,
        "後",
        "/",
        "あと",
        SURF_TOKEN,
        "一",
        READING_TOKEN,
        "ついた",
        "ち",
        LEMMA_TOKEN,
        "一",
        CANON_TOKEN,
        "一",
        "/",
        "いち",
        SURF_TOKEN,
        "日",
        READING_TOKEN,
        FULL_SPACE_TOKEN,
        LEMMA_TOKEN,
        "日",
        CANON_TOKEN,
        "日",
        "/",
        "にち",
        SURF_TOKEN,
        "まで",
        READING_TOKEN,
        "まで",
        LEMMA_TOKEN,
        "まで",
        CANON_TOKEN,
        "まで",
        "/",
        "まで",
        SURF_TOKEN,
        "!",
        READING_TOKEN,
        "!",
        LEMMA_TOKEN,
        "!",
        CANON_TOKEN,
        "!",
        "/",
        "!",
        SURF_TOKEN,
        "?",
        READING_TOKEN,
        "?",
        LEMMA_TOKEN,
        "?",
        CANON_TOKEN,
        "?",
        "/?",
        SURF_TOKEN,
        ".",
        READING_TOKEN,
        ".",
        LEMMA_TOKEN,
        ".",
        CANON_TOKEN,
        ".",
        "/",
        ".",
        SURF_TOKEN,
        "/",
        READING_TOKEN,
        "/",
        LEMMA_TOKEN,
        "/",
        CANON_TOKEN,
        "▁///",
        SURF_TOKEN,
        "°",
        "C",
        READING_TOKEN,
        "ど",
        LEMMA_TOKEN,
        "°",
        "C",
        CANON_TOKEN,
        "°",
        "C",
        "/",
        "ど",
        "</s>",
    ],
    [
        SURF_TOKEN,
        "▁J",
        "UMP",
        READING_TOKEN,
        "▁J",
        "UMP",
        LEMMA_TOKEN,
        "▁J",
        "UMP",
        CANON_TOKEN,
        "▁J",
        "UMP",
        "/",
        "J",
        "UMP",
        SURF_TOKEN,
        FULL_SPACE_TOKEN,
        READING_TOKEN,
        FULL_SPACE_TOKEN,
        LEMMA_TOKEN,
        FULL_SPACE_TOKEN,
        CANON_TOKEN,
        "/",
        SURF_TOKEN,
        "COM",
        "ICS",
        READING_TOKEN,
        "COM",
        "ICS",
        LEMMA_TOKEN,
        "COM",
        "ICS",
        CANON_TOKEN,
        "COM",
        "ICS",
        "/",
        "COM",
        "ICS",
        SURF_TOKEN,
        HALF_SPACE_TOKEN1,
        READING_TOKEN,
        HALF_SPACE_TOKEN1,
        LEMMA_TOKEN,
        HALF_SPACE_TOKEN1,
        CANON_TOKEN,
        "/",
        "</s>",
    ],
    [
        "<pad>",
        "う",
        "ぅ",
        "〜〜",
        READING_TOKEN,
        "<pad>",
        "<pad>",
        "う",
        CANON_TOKEN,
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
    ],
    [
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "さ",
        "き",
        "<pad>",
        "さ",
        "き",
        "<pad>",
        "さ",
        "き",
        "<pad>",
        "先",
        "/",
        "さ",
        "き",
        SURF_TOKEN,
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "かん",
        "<pad>",
        "かん",
        "<pad>",
        "かん",
        "<pad>",
        "館",
        "/",
        "かん",
        "</s>",
    ],
]


texts: List[str] = [
    "計算機による言語理解を実現する",
    "また,校区で行われる事業や防犯など校区の情報も記載されています。",
    f"「核の歴史{TRIPLE_DOT_TOKEN}ヒロシマ、ナガサキを超えて」。",
    "後一日まで!?./°C",
    f"JUMP{FULL_SPACE_TOKEN}COMICS{HALF_SPACE_TOKEN1}",
    "うぅ〜〜お腹が痛い",
    "渡航さきの在日大使かん",
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
        ℃ ど ℃ ℃/ど
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
    dedent(
        """\
        うぅ〜〜 う う 卯/う
        お腹 おなか お腹 お腹/おなか
        が が が が/が
        痛い いたい 痛い 痛い/いたい
        """
    ),
    dedent(
        """\
        渡航 とこう 渡航 渡航/とこう
        さき さき さき 先/さき
        の の の の/の
        在 ざい 在 在/ざい
        日 にち 日 日/にち
        大使 たいし 大使 大使/たいし
        かん かん かん 館/かん
        """
    ),
]

target_morphemes: Dict[str, Dict[str, Dict[str, str]]] = {
    "0": {},
    "1": {},
    "2": {},
    "3": {},
    "4": {},
    "5": {
        "0": {
            "partial_annotation_type": "norm",
            "surf": "うぅ〜〜",
            "lemma": "う",
            "conjtype": "*",
            "conjform": "*",
            "pseudo_canon_type": "",
        },
    },
    "6": {
        "1": {
            "partial_annotation_type": "canon",
            "surf_before": "先",
            "lemma_before": "先",
            "surf_after": "さき",
            "lemma_after": "さき",
            "conjtype": "*",
            "conjform": "*",
            "pseudo_canon_type": "活用なし",
        },
        "6": {
            "partial_annotation_type": "canon",
            "surf_before": "館",
            "lemma_before": "館",
            "surf_after": "かん",
            "lemma_after": "かん",
            "conjtype": "*",
            "conjform": "*",
            "pseudo_canon_type": "活用なし",
        },
    },
}


def test_tokenize(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            mrph_lines: List[List[str]] = seq2seq_formatter.sent_to_mrph_lines(sent)
            tgt_tokens: List[str] = seq2seq_formatter.tokenize(mrph_lines, target_morphemes[str(idx)])
            assert tgt_tokens == tokenizeds[idx]


def test_sent_to_text(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            assert seq2seq_formatter.sent_to_text(sent) == texts[idx]


def test_sent_to_mrph_lines(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sent = Sentence.from_jumanpp(f.read())
            expected_mrph_lines: List[List[str]] = []
            for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
                expected_mrph_lines.append(line.split(" "))
            assert seq2seq_formatter.sent_to_mrph_lines(sent) == expected_mrph_lines


def test_format_to_sent(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            expected_sent = Sentence.from_jumanpp(f.read())

        seq2seq_format: List[str] = []
        for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
            morphemes: List[str] = line.split(" ")
            seq2seq_format.extend(
                f"{SURF_TOKEN} {morphemes[0]} {READING_TOKEN} {morphemes[1]} {LEMMA_TOKEN} {morphemes[2]} {CANON_TOKEN} {morphemes[3]}".split(
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

            actual_canon: str = str(actual_morpheme.canon) if actual_morpheme.canon != NO_CANON_TOKEN else "NIL"
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
