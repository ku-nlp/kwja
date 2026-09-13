import argparse
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def workaround_hydra_argparse() -> Iterator[None]:
    """Work around Hydra's incompatible shell-completion help on Python 3.14."""
    if sys.version_info < (3, 14):
        yield
        return

    # https://github.com/facebookresearch/hydra/issues/3121
    original_check_help: Callable[[argparse.ArgumentParser, argparse.Action], None] = (
        argparse.ArgumentParser._check_help
    )

    def check_help(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.dest == "shell_completion" and not isinstance(action.help, str):
            return
        original_check_help(parser, action)

    argparse.ArgumentParser._check_help = check_help
    try:
        yield
    finally:
        argparse.ArgumentParser._check_help = original_check_help
