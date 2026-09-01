import argparse
import sys

import pytest

from kwja.utils.hydra_compat import workaround_hydra_argparse


@pytest.mark.skipif(sys.version_info < (3, 14), reason="argparse validates help strings from Python 3.14")
def test_workaround_hydra_argparse() -> None:
    class LazyCompletionHelp:
        def __repr__(self) -> str:
            return "shell completion help"

    original_check_help = argparse.ArgumentParser._check_help
    with workaround_hydra_argparse():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--shell-completion", help=LazyCompletionHelp())
        with pytest.raises(ValueError, match="badly formed help string"):
            parser.add_argument("--other", help=LazyCompletionHelp())

    assert argparse.ArgumentParser._check_help is original_check_help
