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
