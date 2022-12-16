import sys
from datetime import timedelta
from functools import partial
from typing import Iterable, List, Optional, Sequence, Union

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, ProgressType, TextColumn
from rich.style import StyleType
from rich.text import Text


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
    update_period=1.0,
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
