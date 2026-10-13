import logging
import warnings
from contextlib import ExitStack
from typing import Literal

import pytest

from kwja.utils.logging_util import filter_logs


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
