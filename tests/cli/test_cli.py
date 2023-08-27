import io
import tempfile
import textwrap
from typing import List, Set, Tuple

import pytest
from rhoknp import Document
from rhoknp.utils.reader import chunk_by_document
from typer.testing import CliRunner

from kwja.cli.cli import app

runner = CliRunner(mix_stderr=False)


def test_version():
    _ = runner.invoke(app, args=["--version"])


def test_device():
    ret = runner.invoke(app, args=["--device", "tpu"])
    assert isinstance(ret.exception, SystemExit)


def test_text_input():
    ret = runner.invoke(app, args=["--model-size", "tiny", "--text", "おはよう"])
    assert ret.exception is None


@pytest.mark.parametrize(
    "text, output",
    [
        ("", "EOD\n"),
        # ("EOD", "EOD\nEOD\n"),  # TODO
        ("おはよう", "おはよう\nEOD\n"),
        ("おはよう．", "おはよう.\nEOD\n"),
        ("おはよう #今日も一日", "おはよう␣＃今日も一日\nEOD\n"),
        ("おはよう。\nこんにちは。\nこんばんわ。\n", "おはよう。こんにちは。こんばんわ。\nEOD\n"),
        ("おはよう。EOD", "おはよう。EOD\nEOD\n"),
    ],
)
def test_normalization_and_typo_module(text: str, output: str):
    ret = runner.invoke(app, args=["--model-size", "tiny", "--tasks", "typo", "--text", text])
    assert ret.exception is None
    assert ret.stdout == output


def test_file_input():
    with tempfile.NamedTemporaryFile("wt") as f:
        f.write(
            textwrap.dedent(
                """\
                KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。
                EOD
                計算機による言語理解を実現するためには，計算機に常識・世界知識を与える必要があります．
                10年前にはこれは非常に難しい問題でしたが，近年の計算機パワー，計算機ネットワークの飛躍的進展によって計算機が超大規模テキストを取り扱えるようになり，そこから常識を自動獲得することが少しずつ可能になってきました．
                EOD
                """
            )
        )
        f.seek(0)
        ret = runner.invoke(app, args=["--model-size", "tiny", "--filename", f.name])
        assert ret.exception is None


def test_sanity():
    with tempfile.NamedTemporaryFile("wt") as f:
        f.write(
            textwrap.dedent(
                """\
                KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。
                EOD
                計算機による言語理解を実現するためには、計算機に常識・世界知識を与える必要があります。
                10年前にはこれは非常に難しい問題でしたが、近年の計算機パワー、計算機ネットワークの飛躍的進展によって
                計算機が超大規模テキストを取り扱えるようになり、そこから常識を自動獲得することが少しずつ可能になってきました。
                EOD
                """
            )
        )
        f.seek(0)
        ret = runner.invoke(app, args=["--model-size", "tiny", "--filename", f.name])
        documents: list[Document] = []
        for knp_text in chunk_by_document(io.StringIO(ret.stdout)):
            documents.append(Document.from_knp(knp_text))
        assert len(documents) == 2
        assert documents[0].text == "KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。"
        assert documents[1].text == (
            "計算機による言語理解を実現するためには、計算機に常識・世界知識を与える必要があります。10年前にはこれは非常に難しい問題でしたが、"
            + "近年の計算機パワー、計算機ネットワークの飛躍的進展によって計算機が超大規模テキストを取り扱えるようになり、そこから常識を"
            + "自動獲得することが少しずつ可能になってきました。"
        )


def test_task_combination_validation():
    tasks: List[str] = [
        "",
        "dummy",
        "typo",
        "typo,char",
        "typo,char,seq2seq",
        "typo,char,word",
        "typo,char,seq2seq,word",
        "char",
        "char,seq2seq",
        "char,word",
        "char,seq2seq,word",
        "seq2seq",
        "seq2seq,word",
        "word",
    ]
    valid_tasks_raw_input: Set[Tuple[str, ...]] = {
        ("typo",),
        ("typo", "char"),
        ("typo", "char", "seq2seq"),
        ("typo", "char", "word"),
        ("typo", "char", "seq2seq", "word"),
        ("char",),
        ("char", "seq2seq"),
        ("char", "word"),
        ("char", "seq2seq", "word"),
    }
    valid_tasks_jumanpp_input: Set[Tuple[str, ...]] = valid_tasks_raw_input | {
        ("seq2seq",),
        ("seq2seq", "word"),
        ("word",),
    }
    base_args: List[str] = ["--model-size", "tiny", "--text", "おはよう"]
    for task in tasks:
        ret = runner.invoke(app, args=base_args + ["--task", task, "--input-format", "raw"])
        if tuple(task.split(",")) not in valid_tasks_raw_input:
            assert isinstance(ret.exception, SystemExit)
        ret = runner.invoke(app, args=base_args + ["--task", task, "--input-format", "jumanpp"])
        if tuple(task.split(",")) not in valid_tasks_jumanpp_input:
            assert isinstance(ret.exception, SystemExit)
