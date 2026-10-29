import logging
import sys
import warnings
from contextlib import ExitStack
from typing import Literal

import pytest
from rich.console import Console

from kwja.utils.logging_util import filter_logs, track


@pytest.mark.parametrize("environment", ["production", "development"])
def test_filter_logs(environment: Literal["production", "development"], caplog: pytest.LogCaptureFixture) -> None:
    with ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings())
        for name in ["kwja", "rhoknp", "transformers", "torch", "lightning", "lightning.pytorch.utilities.rank_zero"]:
            stack.enter_context(caplog.at_level(logging.DEBUG, logger=name))
        filter_logs(environment=environment)
        logger = logging.getLogger("kwja.utils.seq2seq_format")
        logger.debug("Internal diagnostic")
        logger.info("Normal operation")
        logger.warning("Partial result")

    expected_messages = ["Partial result"]
    if environment == "development":
        expected_messages.insert(0, "Normal operation")
    assert caplog.messages == expected_messages


@pytest.mark.parametrize("quiet", [True, False])
def test_track_preserves_stderr(quiet: bool, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(file=sys.stderr, quiet=quiet)
    values = []
    for value in track([1, 2], console=console):
        values.append(value)
        print("Warning during iteration", file=sys.stderr)
    assert values == [1, 2]
    stderr = capsys.readouterr().err
    assert stderr.count("Warning during iteration") == 2
    assert ("Working..." in stderr) == (not quiet)
