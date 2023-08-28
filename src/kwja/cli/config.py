import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

ENV_KWJA_CONFIG_FILE = "KWJA_CONFIG_FILE"
ENV_XDG_CONFIG_HOME = "XDG_CONFIG_HOME"
DEFAULT_CONFIG_DIR = Path.home() / ".config"


def get_kwja_config_file() -> Path:
    if path := os.getenv(ENV_KWJA_CONFIG_FILE):
        return Path(path)
    config_dir = DEFAULT_CONFIG_DIR
    if path := os.getenv(ENV_XDG_CONFIG_HOME):
        config_dir = Path(path)
    return config_dir / "kwja" / "config.yaml"


class Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class ModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    LARGE = "large"


@dataclass
class CLIConfig:
    model_size: ModelSize = ModelSize.BASE
    device: Device = Device.AUTO
    num_workers: int = 0
    torch_compile: bool = False
    typo_batch_size: int = 1
    char_batch_size: int = 1
    seq2seq_batch_size: int = 1
    word_batch_size: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> "CLIConfig":
        config = yaml.safe_load(path.read_text())
        return cls(
            model_size=ModelSize(config["model_size"]),
            device=Device(config["device"]),
            num_workers=config["num_workers"],
            torch_compile=config["torch_compile"],
            typo_batch_size=config["typo_batch_size"],
            char_batch_size=config["char_batch_size"],
            seq2seq_batch_size=config["seq2seq_batch_size"],
            word_batch_size=config["word_batch_size"],
        )
