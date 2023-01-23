import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import torch
import transformers.utils.logging as hf_logging
from torch.hub import download_url_to_file

import kwja

ENV_KWJA_CACHE_DIR = "KWJA_CACHE_DIR"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = Path.home() / ".cache"

_CHECKPOINT_BASE_URL = "https://lotus.kuee.kyoto-u.ac.jp"
_CHECKPOINT_FILE_NAMES = {
    "tiny": {
        "typo": "typo_deberta-v2-tiny-wwm.ckpt",
        "char": "char_deberta-v2-tiny-wwm.ckpt",
        "word": "word_deberta-v2-tiny.ckpt",
        "word_discourse": "disc_deberta-v2-tiny.ckpt",
    },
    "base": {
        "typo": "typo_roberta-base-wwm.ckpt",
        "char": "char_roberta-base-wwm.ckpt",
        "word": "word_roberta-base.ckpt",
        "word_discourse": "disc_roberta-base.ckpt",
    },
    "large": {
        "typo": "typo_roberta-large-wwm.ckpt",
        "char": "char_roberta-large-wwm.ckpt",
        "word": "word_roberta-large.ckpt",
        "word_discourse": "disc_roberta-large.ckpt",
    },
}

logger = logging.getLogger(__name__)


def suppress_debug_info() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("kwja").setLevel(logging.ERROR)
    logging.getLogger("rhoknp").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    hf_logging.set_verbosity(hf_logging.ERROR)


def download_checkpoint(
    task: str,
    model_size: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Path:
    """Downloads the Torch serialized object at the given URL.
    If the object is already present in `checkpoint_dir`, just return the path to the object.

    Args:
        task: typo, char, word, or word_discourse
        model_size: base or large
        checkpoint_dir: directory in which to save the object
        progress: whether to display a progress bar to stderr
    """

    if checkpoint_dir is None:
        checkpoint_dir = _get_kwja_cache_dir() / _get_model_version()
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    remote_checkpoint_path = Path("/kwja") / _get_model_version() / _CHECKPOINT_FILE_NAMES[model_size][task]
    checkpoint_url = _CHECKPOINT_BASE_URL + str(remote_checkpoint_path)
    parts = urlparse(checkpoint_url)
    filename = os.path.basename(parts.path)
    checkpoint_path = checkpoint_dir / filename
    if checkpoint_path.exists() is False:
        sys.stderr.write(f'Downloading: "{checkpoint_url}" to {checkpoint_path}\n')
        download_url_to_file(checkpoint_url, str(checkpoint_path), None, progress=progress)
    return checkpoint_path


def _get_kwja_cache_dir() -> Path:
    if path := os.getenv(ENV_KWJA_CACHE_DIR):
        return Path(path)
    cache_dir = DEFAULT_CACHE_DIR
    if path := os.getenv(ENV_XDG_CACHE_HOME):
        cache_dir = Path(path)
    return cache_dir / "kwja"


def _get_model_version() -> str:
    version_map = {
        ("1", "0"): "v1.0",
        ("1", "1"): "v1.0",
        ("1", "2"): "v1.0",
        ("1", "3"): "v1.3",
        # ("1", "4"): "v1.3",
    }
    version_fields = kwja.__version__.split(".")
    return version_map[(version_fields[0], version_fields[1])]


def prepare_device(device_name: str) -> Tuple[str, torch.device]:
    n_gpu = torch.cuda.device_count()
    if device_name == "auto":
        if n_gpu > 0:
            device_name = "gpu"
        else:
            device_name = "cpu"
    if device_name == "gpu" and n_gpu == 0:
        logger.warning("There's no GPU available on this machine. Using CPU instead.")
        return "cpu", torch.device("cpu")
    else:
        return device_name, torch.device("cuda:0" if device_name == "gpu" else "cpu")
