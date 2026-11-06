import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, TextIO

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter


class BaseModuleWriter(BasePredictionWriter):
    def __init__(self, destination: str | Path | None = None) -> None:
        super().__init__(write_interval="batch")
        if destination is None:
            self.destination: Path | TextIO = sys.stdout
        else:
            if isinstance(destination, str):
                destination = Path(destination)
            self.destination = destination
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            self.destination.unlink(missing_ok=True)

    def write_output_string(self, output_string: str) -> None:
        if isinstance(self.destination, Path):
            with self.destination.open(mode="a", encoding="utf-8") as f:
                f.write(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    @staticmethod
    def get_predict_dataloader(trainer: L.Trainer, dataloader_idx: int) -> Any:
        predict_dataloaders = trainer.predict_dataloaders
        assert predict_dataloaders is not None  # noqa: S101
        if isinstance(predict_dataloaders, dict):
            return list(predict_dataloaders.values())[dataloader_idx]
        return predict_dataloaders[dataloader_idx]
