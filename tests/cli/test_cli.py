from typing import List

from typer.testing import CliRunner

from kwja.cli.cli import app

runner = CliRunner()


def test_version():
    _ = runner.invoke(app, args=["--version"])


def test_device():
    ret = runner.invoke(app, args=["--device", "tpu"])
    assert isinstance(ret.exception, SystemExit)


def test_text_input():
    _ = runner.invoke(app, args=["--model-size", "tiny", "--text", "おはよう"])


def test_file_input():
    _ = runner.invoke(app, args=["--model-size", "tiny", "--filename", "./sample.txt"])


def test_task_input():
    tasks: List[str] = [
        "",
        "dummy",
        "typo",
        "char",
        "word",
        "word_discourse",
        "typo,char",
        "typo,word",
        "typo,word_discourse",
        "char,word",
        "char,word_discourse",
        "word,word_discourse",
        "typo,char,word",
        "typo,char,word_discourse",
        "typo,word,word_discourse",
        "char,word,word_discourse",
        "typo,char,word,word_discourse",
    ]
    valid_tasks: List[str] = [
        "typo",
        "char",
        "typo,char",
        "char,word",
        "typo,char,word",
        "char,word,word_discourse",
        "typo,char,word,word_discourse",
    ]
    for task in tasks:
        ret = runner.invoke(app, args=["--model-size", "tiny", "--text", "おはよう", "--task", task])
        if task not in valid_tasks:
            assert isinstance(ret.exception, SystemExit)
