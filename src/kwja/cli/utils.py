import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch.hub import download_url_to_file

import kwja
from kwja.cli.config import ModelSize

logger = logging.getLogger(__name__)

ENV_KWJA_CACHE_DIR = "KWJA_CACHE_DIR"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = Path.home() / ".cache"

_CHECKPOINT_BASE_URL = "https://lotus.kuee.kyoto-u.ac.jp"
_CHECKPOINT_FILE_NAMES: Dict[ModelSize, Dict[str, str]] = {
    ModelSize.TINY: {
        "typo": "typo_deberta-v2-tiny-wwm.ckpt",
        "char": "char_deberta-v2-tiny-wwm.ckpt",
        "seq2seq": "seq2seq_t5-small.ckpt",
        "word": "word_deberta-v2-tiny.ckpt",
    },
    ModelSize.BASE: {
        "typo": "typo_deberta-v2-base-wwm.ckpt",
        "char": "char_deberta-v2-base-wwm.ckpt",
        "seq2seq": "seq2seq_t5-base.ckpt",
        "word": "word_deberta-v2-base.ckpt",
    },
    ModelSize.LARGE: {
        "typo": "typo_deberta-v2-large-wwm.ckpt",
        "char": "char_deberta-v2-large-wwm.ckpt",
        "seq2seq": "seq2seq_t5-large.ckpt",
        "word": "word_deberta-v2-large.ckpt",
    },
}


def download_checkpoint(
    module: str,
    model_size: ModelSize,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Path:
    """Downloads the Torch serialized object at the given URL.
    If the object is already present in `checkpoint_dir`, just return the path to the object.

    Args:
        module: typo, char, or word
        model_size: tiny, base or large
        checkpoint_dir: directory in which to save the object
        progress: whether to display a progress bar to stderr
    """

    if checkpoint_dir is None:
        checkpoint_dir = _get_kwja_cache_dir() / _get_model_version()
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_url = f"{_CHECKPOINT_BASE_URL}/kwja/{_get_model_version()}/{_CHECKPOINT_FILE_NAMES[model_size][module]}"
    checkpoint_path = checkpoint_dir / _CHECKPOINT_FILE_NAMES[model_size][module]
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
    if "dev" in kwja.__version__:
        return "dev"
    version_map = {
        ("1", "0"): "v1.0",
        ("1", "1"): "v1.0",
        ("1", "2"): "v1.0",
        ("1", "3"): "v1.3",
        ("1", "4"): "v1.3",
        ("2", "0"): "v2.0",
        ("2", "1"): "v2.1",
        ("2", "2"): "v2.2",
        ("2", "3"): "v2.2",
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
