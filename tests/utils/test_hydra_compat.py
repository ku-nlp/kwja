import argparse
import sys
from unittest.mock import patch

import pytest

from kwja.utils.hydra_compat import workaround_hydra_argparse


@pytest.mark.skipif(sys.version_info < (3, 14), reason="argparse validates help strings from Python 3.14")
def test_workaround_hydra_argparse() -> None:
    class LazyCompletionHelp:
        def __repr__(self) -> str:
            return "shell completion help"

    original_check_help = argparse.ArgumentParser._check_help
    with patch.object(
        argparse.ArgumentParser, "_check_help", autospec=True, side_effect=original_check_help
    ) as check_help:
        with workaround_hydra_argparse():
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--shell-completion", help=LazyCompletionHelp())
            check_help.assert_not_called()
            action = parser.add_argument("--other", help="other help")
            check_help.assert_called_once_with(parser, action)
        assert argparse.ArgumentParser._check_help is check_help

    assert argparse.ArgumentParser._check_help is original_check_help
