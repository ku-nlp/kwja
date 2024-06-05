from pathlib import Path
from textwrap import dedent
from typing import List

import jaconv
from rhoknp import Sentence
from transformers import PreTrainedTokenizerFast

from kwja.utils.constants import (
    CANON_TOKEN,
    HALF_SPACE_TOKEN,
    LEMMA_TOKEN,
    MORPHEME_SPLIT_TOKEN,
    NO_CANON_TOKEN,
    READING_TOKEN,
    SURF_TOKEN,
)
from kwja.utils.normalization import normalize_morpheme
from kwja.utils.seq2seq_format import Seq2SeqFormatter

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
        ... ... ... .../...
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
        日 {HALF_SPACE_TOKEN} 日 日/にち
        まで まで まで まで/まで
        ！ ！ ！ ！/！
        ？ ？ ？ ？/？
        . . . ./.
        / / / ///
        °C ど °C °C/ど
        """
    ),
    dedent(
        f"""\
        JUMP JUMP JUMP JUMP/JUMP
        {HALF_SPACE_TOKEN} {HALF_SPACE_TOKEN} {HALF_SPACE_TOKEN} /
        COMICS COMICS COMICS COMICS/COMICS
        {HALF_SPACE_TOKEN} {HALF_SPACE_TOKEN} {HALF_SPACE_TOKEN} /
        """
    ),
]


def test_get_surfs(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    surfs: List[str] = [
        "計算 機 に よる 言語 理解 を 実現 する",
        "また , 校区 で 行わ れる 事業 や 防犯 など 校区 の 情報 も 記載 さ れて い ます 。",
        "「 核 の 歴史 ... ヒロシマ 、 ナガサキ を 超えて 」 。",
        "後 一 日 まで ! ? . / °C",
        f"JUMP {HALF_SPACE_TOKEN} COMICS {HALF_SPACE_TOKEN}",
    ]
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sentence = Sentence.from_jumanpp(f.read())
            for morpheme in sentence.morphemes:
                normalize_morpheme(morpheme)
            assert seq2seq_formatter.get_surfs(sentence) == surfs[idx].split(" ")


def test_get_src_tokens(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    src_tokens: List[List[str]] = [
        ["計算", "機", "に", "よ る", "言語", "理解", "を", "実現", "する"],
        [
            "また",
            ",",
            "校 区",
            "で",
            "行 わ",
            "れる",
            "事業",
            "や",
            "防犯",
            "など",
            "校 区",
            "の",
            "情報",
            "も",
            "記載",
            "さ",
            "れて",
            "い",
            "ます",
            "。",
        ],
        ["「", "核", "の", "歴史", "...", "ヒロ シマ", "、", "ナ ガ サ キ", "を", "超 えて", "」", "。"],
        ["後", "一", "日", "まで", "!", "?", ".", "/", "° C"],
        ["J UMP", HALF_SPACE_TOKEN, "COM ICS", HALF_SPACE_TOKEN],
    ]
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sentence = Sentence.from_jumanpp(f.read())
            for morpheme in sentence.morphemes:
                normalize_morpheme(morpheme)
            expected_src_tokens: List[str] = []
            for morphemes in src_tokens[idx]:
                expected_src_tokens.extend([*morphemes.split(" "), MORPHEME_SPLIT_TOKEN])
            assert [
                x[1:] if x.startswith("▁") else x for x in seq2seq_formatter.get_src_tokens(sentence)
            ] == expected_src_tokens[:-1]


def test_get_tgt_tokens(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            sentence = Sentence.from_jumanpp(f.read())
            for morpheme in sentence.morphemes:
                normalize_morpheme(morpheme)
            expected_seq2seq_format: str = ""
            for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
                mrphs: List[str] = line.split(" ")
                expected_seq2seq_format += (
                    f"{SURF_TOKEN}{mrphs[0]}{READING_TOKEN}{mrphs[1]}{LEMMA_TOKEN}{mrphs[2]}{CANON_TOKEN}{mrphs[3]}"
                )
            expected_tgt_tokens: List[str] = [
                x for x in seq2seq_tokenizer.tokenize(expected_seq2seq_format) if x != "▁"
            ]
            assert seq2seq_formatter.get_tgt_tokens(sentence) == expected_tgt_tokens


def test_format_to_sent(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    seq2seq_formatter = Seq2SeqFormatter(seq2seq_tokenizer)
    juman_dir: Path = data_dir / "modules" / "juman"
    for idx, path in enumerate(sorted(juman_dir.glob("*.juman"))):
        with open(path) as f:
            expected_sentence = Sentence.from_jumanpp(f.read())
            for morpheme in expected_sentence.morphemes:
                normalize_morpheme(morpheme)

        seq2seq_format: List[str] = []
        for line in seq2seq_formats[idx].rstrip("\n").split("\n"):
            morphemes: List[str] = line.split(" ")
            seq2seq_format.extend(
                f"{SURF_TOKEN} {morphemes[0]} {READING_TOKEN} {morphemes[1]} {LEMMA_TOKEN} {morphemes[2]} {CANON_TOKEN} {morphemes[3]}".split(
                    " "
                )
            )

        actual_sentence: Sentence = seq2seq_formatter.format_to_sent("".join(seq2seq_format))
        assert len(actual_sentence.morphemes) == len(expected_sentence.morphemes)
        for actual_morpheme, expected_morpheme in zip(actual_sentence.morphemes, expected_sentence.morphemes):
            if "/" in expected_morpheme.reading and len(expected_morpheme.reading) > 1:
                expected_reading: str = expected_morpheme.reading.split("/")[0]
            else:
                expected_reading = expected_morpheme.reading

            if actual_morpheme.canon == r"\␣/\␣":
                actual_canon: str = "/"
            else:
                actual_canon = str(actual_morpheme.canon) if actual_morpheme.canon != NO_CANON_TOKEN else "NIL"
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
