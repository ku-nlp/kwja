import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import transformers.utils.logging as hf_logging
from torch.hub import download_url_to_file

ENV_KWJA_CACHE_DIR = "KWJA_CACHE_DIR"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = Path.home() / ".cache"


def suppress_debug_info() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("kwja").setLevel(logging.ERROR)
    logging.getLogger("rhoknp").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    hf_logging.set_verbosity(hf_logging.ERROR)


def download_checkpoint_from_url(
    url: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Path:
    """Downloads the Torch serialized object at the given URL.
    If the object is already present in `checkpoint_dir`, just return the path to the object.

    Args:
        url: URL of the object to download
        checkpoint_dir: directory in which to save the object
        progress: whether to display a progress bar to stderr
    """

    if checkpoint_dir is None:
        checkpoint_dir = _get_kwja_cache_dir()
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    checkpoint_path = checkpoint_dir / filename
    if checkpoint_path.exists() is False:
        sys.stderr.write(f'Downloading: "{url}" to {checkpoint_path}\n')
        download_url_to_file(url, str(checkpoint_path), None, progress=progress)
    return checkpoint_path


def _get_kwja_cache_dir() -> Path:
    if path := os.getenv(ENV_KWJA_CACHE_DIR):
        return Path(path)
    cache_dir = DEFAULT_CACHE_DIR
    if path := os.getenv(ENV_XDG_CACHE_HOME):
        cache_dir = Path(path)
    return cache_dir / "kwja"
