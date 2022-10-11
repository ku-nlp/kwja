from typer.testing import CliRunner

from kwja.cli.cli import app

runner = CliRunner()


def test_version():
    _ = runner.invoke(app, args=["--version"])


# def test_text_input():
#     _ = runner.invoke(app, args=["--text", "おはよう"])
#
#
# def test_file_input():
#     _ = runner.invoke(app, args=["--filename", "./sample.txt"])
