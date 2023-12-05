import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter


class BaseModuleWriter(BasePredictionWriter):
    def __init__(self, destination: Optional[Union[str, Path]] = None) -> None:
        super().__init__(write_interval="batch")
        if destination is None:
            self.destination: Union[Path, TextIO] = sys.stdout
        else:
            if isinstance(destination, str):
                destination = Path(destination)
            self.destination = destination
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            self.destination.unlink(missing_ok=True)

    def write_output_string(self, output_string: str) -> None:
        if isinstance(self.destination, Path):
            with self.destination.open(mode="a") as f:
                f.write(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        raise NotImplementedError

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        pass  # pragma: no cover
