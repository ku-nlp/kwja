import io
import re
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
    ret = runner.invoke(app, args=["--version"])
    assert re.match(r"KWJA \d+\.\d+\.\d+", ret.stdout) is not None


def test_device_validation():
    devices: List[str] = ["", "auto", "cpu", "gpu", "INVALID"]
    base_args: List[str] = ["--model-size", "tiny", "--text", ""]
    for device in devices:
        ret = runner.invoke(app, args=[*base_args, "--device", device])
        if device not in {"auto", "cpu", "gpu"}:
            assert isinstance(ret.exception, SystemExit)
        else:
            assert ret.exception is None


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
    base_args: List[str] = ["--model-size", "tiny", "--text", ""]
    for task in tasks:
        ret = runner.invoke(app, args=[*base_args, "--task", task, "--input-format", "raw"])
        if tuple(task.split(",")) not in valid_tasks_raw_input:
            assert isinstance(ret.exception, SystemExit)
        ret = runner.invoke(app, args=[*base_args, "--task", task, "--input-format", "jumanpp"])
        if tuple(task.split(",")) not in valid_tasks_jumanpp_input:
            assert isinstance(ret.exception, SystemExit)


def test_input_format_validation():
    input_formats: List[str] = ["", "raw", "jumanpp", "knp", "INVALID"]
    base_args: List[str] = ["--model-size", "tiny", "--text", ""]
    for input_format in input_formats:
        ret = runner.invoke(app, args=[*base_args, "--input-format", input_format])
        if input_format not in {"raw", "jumanpp", "knp"}:
            assert isinstance(ret.exception, SystemExit)
        else:
            assert ret.exception is None


def test_text_input():
    ret = runner.invoke(app, args=["--model-size", "tiny", "--text", "おはよう"])
    assert ret.exception is None
    assert Document.from_knp(ret.stdout).text == "おはよう"


@pytest.mark.parametrize(
    ("text", "output"),
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


@pytest.mark.parametrize(
    ("text", "output"),
    [
        ("今日は${day}日です", "今日は${day}日です"),
    ],
)
def test_normalization_and_char_module(text: str, output: str):
    ret = runner.invoke(app, args=["--model-size", "tiny", "--tasks", "char", "--text", text])
    assert ret.exception is None
    restored_output = "".join([line for line in ret.stdout.splitlines() if not line.startswith("#")]).replace(" ", "")
    assert restored_output == output.replace(" ", "")


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


@pytest.mark.parametrize(
    "text",
    [
        "EOD\n",
        "おはよう\nEOD\n",
        "KWJAは日本語の統合解析ツールです。\n汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。\nEOD\n",
    ],
)
def test_interactive_mode(text: str):
    ret = runner.invoke(app, args=["--model-size", "tiny", "--tasks", "char,word"], input=text)
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
        assert (
            documents[0].text
            == "KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。"
        )
        assert documents[1].text == (
            "計算機による言語理解を実現するためには、計算機に常識・世界知識を与える必要があります。10年前にはこれは非常に難しい問題でしたが、"
            + "近年の計算機パワー、計算機ネットワークの飛躍的進展によって計算機が超大規模テキストを取り扱えるようになり、そこから常識を"
            + "自動獲得することが少しずつ可能になってきました。"
        )


@pytest.mark.parametrize(
    "tasks",
    [
        "word",
        "seq2seq",
        "char,word",
        "char,seq2seq",
        "seq2seq,word",
        "typo,char,word",
        "typo,char,seq2seq",
        "char,seq2seq,word",
        "typo,char,seq2seq,word",
    ],
)
def test_input_format_jumanpp(tasks: str):
    jumanpp_text = textwrap.dedent(
        """\
        こんにちは こんにちは こんにちは 感動詞 12 * 0 * 0 * 0 "代表表記:こんにちは/こんにちは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        こんばんは こんばんは こんばんは 感動詞 12 * 0 * 0 * 0 "代表表記:今晩は/こんばんは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """
    )
    ret = runner.invoke(
        app, args=["--model-size", "tiny", "--tasks", tasks, "--text", jumanpp_text, "--input-format", "jumanpp"]
    )
    assert ret.exception is None
    documents: List[Document] = []
    for out_text in chunk_by_document(io.StringIO(ret.stdout)):
        if tasks.split(",")[-1] == "word":
            documents.append(Document.from_knp(out_text))
        else:
            documents.append(Document.from_jumanpp(out_text))
    assert len(documents) == 1
    document = documents[0]
    doc_id = document.doc_id
    assert doc_id != ""  # current datetime
    sentences = document.sentences
    for sentence in sentences:
        assert sentence.sent_id.startswith(doc_id)
        assert sentence.doc_id == doc_id
    if "typo" not in tasks:
        assert document.text == "こんにちは。こんばんは。"
    if tasks.split(",")[0] in ("word", "seq2seq"):
        # sentence boundaries are kept
        assert len(sentences) == 2
        assert sentences[0].text == "こんにちは。"
        assert sentences[0].sent_id.startswith(doc_id)
        assert sentences[1].text == "こんばんは。"
        assert sentences[1].sent_id.startswith(doc_id)


@pytest.mark.parametrize(
    "tasks",
    [
        "word",
        "seq2seq",
        "char,word",
        "char,seq2seq",
        "seq2seq,word",
        "typo,char,word",
        "typo,char,seq2seq",
        "char,seq2seq,word",
        "typo,char,seq2seq,word",
    ],
)
def test_input_format_jumanpp_with_sid(tasks: str):
    jumanpp_text = textwrap.dedent(
        """\
        # S-ID:test-0 kwja:0.1.0
        こんにちは こんにちは こんにちは 感動詞 12 * 0 * 0 * 0 "代表表記:こんにちは/こんにちは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        # S-ID:test-1 kwja:0.1.0
        こんばんは こんばんは こんばんは 感動詞 12 * 0 * 0 * 0 "代表表記:今晩は/こんばんは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """
    )
    ret = runner.invoke(
        app, args=["--model-size", "tiny", "--tasks", tasks, "--text", jumanpp_text, "--input-format", "jumanpp"]
    )
    assert ret.exception is None
    documents: List[Document] = []
    for out_text in chunk_by_document(io.StringIO(ret.stdout)):
        if tasks.split(",")[-1] == "word":
            documents.append(Document.from_knp(out_text))
        else:
            documents.append(Document.from_jumanpp(out_text))
    assert len(documents) == 1
    document = documents[0]
    doc_id = document.doc_id
    assert doc_id == "test"
    sentences = document.sentences
    for sentence in sentences:
        assert sentence.sent_id.startswith(doc_id)
        assert sentence.doc_id == doc_id
    if "typo" not in tasks:
        assert document.text == "こんにちは。こんばんは。"
    if tasks.split(",")[0] in ("word", "seq2seq"):
        # sentence boundaries are kept
        assert len(sentences) == 2
        assert sentences[0].text == "こんにちは。"
        assert sentences[0].sent_id == "test-0"
        assert sentences[1].text == "こんばんは。"
        assert sentences[1].sent_id == "test-1"


@pytest.mark.parametrize(
    "tasks",
    [
        "word",
        "seq2seq",
        "char,word",
        "char,seq2seq",
        "seq2seq,word",
        "typo,char,word",
        "typo,char,seq2seq",
        "char,seq2seq,word",
        "typo,char,seq2seq,word",
    ],
)
def test_input_format_knp(tasks: str):
    knp_text = textwrap.dedent(
        """\
        # S-ID:test-0 kwja:0.1.0
        * -1D
        + -1D
        こんにちは こんにちは こんにちは 感動詞 12 * 0 * 0 * 0 "代表表記:こんにちは/こんにちは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        # S-ID:test-1 kwja:0.1.0
        * -1D
        + -1D
        こんばんは こんばんは こんばんは 感動詞 12 * 0 * 0 * 0 "代表表記:今晩は/こんばんは"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """
    )
    ret = runner.invoke(
        app, args=["--model-size", "tiny", "--tasks", tasks, "--text", knp_text, "--input-format", "knp"]
    )
    assert ret.exception is None
    documents: List[Document] = []
    for out_text in chunk_by_document(io.StringIO(ret.stdout)):
        if tasks.split(",")[-1] == "word":
            documents.append(Document.from_knp(out_text))
        else:
            documents.append(Document.from_jumanpp(out_text))
    assert len(documents) == 1
    document = documents[0]
    doc_id = document.doc_id
    assert doc_id == "test"
    sentences = document.sentences
    for sentence in sentences:
        assert sentence.sent_id.startswith(doc_id)
        assert sentence.doc_id == doc_id
    if "typo" not in tasks:
        assert documents[0].text == "こんにちは。こんばんは。"
    if tasks.split(",")[0] in ("word", "seq2seq"):
        # sentence boundaries are kept
        assert len(sentences) == 2
        assert sentences[0].text == "こんにちは。"
        assert sentences[0].sent_id == "test-0"
        assert sentences[1].text == "こんばんは。"
        assert sentences[1].sent_id == "test-1"
