from typing import List, Set, Tuple

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
        "typo,senter",
        "typo,senter,char",
        "typo,senter,word",
        "typo,senter,word_discourse",
        "typo,senter,seq2seq",
        "typo,senter,seq2seq,char",
        "typo,senter,seq2seq,word",
        "typo,senter,seq2seq,word_discourse",
        "senter",
        "senter,char",
        "senter,char,word",
        "senter,char,word,word_discourse",
        "senter,seq2seq",
        "senter,seq2seq,word",
        "senter,seq2seq,word,word_discourse",
    ]
    valid_tasks: Set[Tuple[str, ...]] = {
        ("typo",),
        ("typo", "senter"),
        ("typo", "senter", "char"),
        ("typo", "senter", "char", "word"),
        ("typo", "senter", "char", "word", "word_discourse"),
        ("typo", "senter", "seq2seq"),
        ("typo", "senter", "seq2seq", "word"),
        ("typo", "senter", "seq2seq", "word", "word_discourse"),
        ("senter",),
        ("senter", "char"),
        ("senter", "char", "word"),
        ("senter", "char", "word", "word_discourse"),
        ("senter", "seq2seq"),
        ("senter", "seq2seq", "word"),
        ("senter", "seq2seq", "word", "word_discourse"),
    }
    for task in tasks:
        ret = runner.invoke(app, args=["--model-size", "tiny", "--text", "おはよう", "--task", task])
        if tuple(task.split(",")) not in valid_tasks:
            assert isinstance(ret.exception, SystemExit)
