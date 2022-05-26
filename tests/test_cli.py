from typer.testing import CliRunner

from jula.cli import app

runner = CliRunner()


def test_app():
    inp = "test"
    result = runner.invoke(app, input=inp)
    assert result.exit_code == 0
    assert result.output == inp + "\n"
