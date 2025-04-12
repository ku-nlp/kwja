import subprocess
from pathlib import Path

import pytest

from kwja.utils import reading_prediction
from kwja.utils.constants import RESOURCE_TRAVERSABLE


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
):
    assert reading_prediction.get_word_level_readings(readings, tokens, subword_map) == expected_output


def test_main(data_dir: Path):
    script_path = reading_prediction.__file__
    assert script_path is not None
    kanjidic_path = RESOURCE_TRAVERSABLE / "reading_prediction" / "kanjidic"
    input_path = data_dir / "datasets" / "word_files"
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            script_path,
            "-m",
            "ku-nlp/deberta-v2-tiny-japanese",
            "-k",
            str(kanjidic_path),
            "-i",
            str(input_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            script_path,
            "-m",
            "nlp-waseda/roberta-base-japanese",
            "-k",
            str(kanjidic_path),
            "-i",
            str(input_path),
        ],
        check=True,
    )
