import logging
import sys
import warnings
from datetime import timedelta
from functools import partial
from typing import Iterable, List, Literal, Optional, Sequence, Union

from lightning_fabric.utilities.warnings import PossibleUserWarning
from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, ProgressType, TextColumn
from rich.style import StyleType
from rich.text import Text
from transformers.utils import logging as hf_logging


def filter_logs(environment: Literal["development", "production"]) -> None:
    logging.getLogger("rhoknp").setLevel(logging.ERROR)
    hf_logging.set_verbosity(hf_logging.ERROR)
    if environment == "production":
        warnings.filterwarnings("ignore")
        logging.getLogger("kwja").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    elif environment == "development":
        warnings.filterwarnings(
            "ignore",
            message=(
                r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
                r" across devices"
            ),
            category=PossibleUserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Using `DistributedSampler` with the dataloaders. During `trainer..+`, it is recommended to use"
                r" `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise,"
                r" multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have"
                r" same batch size in case of uneven inputs."
            ),
            category=PossibleUserWarning,
        )


class CustomPostfixColumn(ProgressColumn):
    def __init__(self, style: Union[str, StyleType]) -> None:
        self.style = style
        super().__init__()

    def render(self, task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))

        elapsed = task.finished_time if task.finished else task.elapsed
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining = task.time_remaining
        remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))

        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(
            f"{completed:{total_width}d}/{total} " f"{elapsed_delta} • {remaining_delta} " f"{task_speed}it/s",
            style=self.style,
        )


def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    console: Optional[Console] = None,
    update_period: float = 1.0,
):
    columns: List[ProgressColumn] = [
        TextColumn("[progress.description]{task.description}", style="white"),
        BarColumn(
            style="bar.back",
            complete_style="#6206E0",
            finished_style="#6206E0",
            pulse_style="#6206E0",
        ),
        CustomPostfixColumn(style="white"),
    ]
    progress = Progress(
        *columns,
        console=console,
        refresh_per_second=1,
    )
    with progress:
        yield from progress.track(sequence, total=total, description=description, update_period=update_period)


CONSOLE = Console(file=sys.stderr)
track = partial(track, console=CONSOLE)
