import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rhoknp import Document
from transformers import PreTrainedTokenizerBase

from kwja.utils import reading_prediction
from kwja.utils.constants import RESOURCE_TRAVERSABLE
from kwja.utils.kanjidic import KanjiDic
from kwja.utils.reading_prediction import ReadingAligner
from kwja.utils.tokenization import tokenize_morphemes


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_align_existing_encoding(word_tokenizer: PreTrainedTokenizerBase, add_special_tokens: bool) -> None:
    document = Document.from_knp(
        "# S-ID:test-0\n* -1D\n+ -1D\n"
        "３ さん ３ 名詞 6 数詞 7 * 0 * 0\n"
        "けんか けんか けんか 名詞 6 サ変名詞 2 * 0 * 0\nEOS\n"
    )
    original_knp = document.to_knp()
    encoding = tokenize_morphemes(document.morphemes, word_tokenizer, add_special_tokens=add_special_tokens)
    aligner = ReadingAligner(word_tokenizer, KanjiDic(RESOURCE_TRAVERSABLE / "reading_prediction" / "kanjidic"))
    with patch.object(type(word_tokenizer), "__call__", side_effect=AssertionError("Unexpected retokenization")):
        readings = aligner.align(document.morphemes, encoding)
    assert readings == ["さん", "けん", "か"]
    assert len(readings) == sum(word_id is not None for word_id in encoding.word_ids)
    assert document.to_knp() == original_knp


@pytest.mark.parametrize(
    ("readings", "tokens", "subword_map", "expected_output"),
    [
        (
            ["[UNK]", "せい", "じょう", "[ID]", "にゅうりょく"],
            ["[CLS]", "正", "常", "な", "入力"],
            [
                [False, True, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            ["せいじょう", "な", "にゅうりょく"],
        ),
        (
            ["[UNK]", "ふせい", "[ID]", "[ID]", "にゅうりょく"],
            ["[CLS]", "不正", "", "な", "入力"],
            [
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ],
            ["ふせい", "_", "な", "にゅうりょく"],
        ),
        (
            ["[UNK]", "ふせい", "[ID]", "[ID]", "にゅうりょく"],
            ["[CLS]", "不正", "", "な", "入力"],
            [
                [False, True, False, False, False],
                [False, False, True, True, False],
                [False, False, False, False, True],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            ["ふせい", "な", "にゅうりょく"],
        ),
    ],
)
def test_get_word_level_readings(
    readings: list[str],
    tokens: list[str],
    subword_map: list[list[bool]],
    expected_output: list[str],
) -> None:
    assert reading_prediction.get_word_level_readings(readings, tokens, subword_map) == expected_output


def test_main(data_dir: Path) -> None:
    script_path = Path(__file__).parents[2] / "scripts" / "preprocessors" / "preprocess_reading.py"
    kanjidic_path = RESOURCE_TRAVERSABLE / "reading_prediction" / "kanjidic"
    input_path = data_dir / "datasets" / "word_files"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script_path),
            "-m",
            "ku-nlp/deberta-v2-tiny-japanese",
            "-k",
            str(kanjidic_path),
            "-i",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script_path),
            "-m",
            "ku-nlp/deberta-v2-base-japanese",
            "-k",
            str(kanjidic_path),
            "-i",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()
