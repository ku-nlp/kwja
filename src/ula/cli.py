import sys

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    inp = sys.stdin.read()
    typer.echo(inp)
