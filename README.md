# jula: A unified language analyzer for Japanese

[![test](https://github.com/ku-nlp/jula/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/jula/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/jula/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/jula)

## Requirements

- Python: 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Installation

To prepare the Python environment of our project, we do:
```shell
poetry install
```
You must prepare config files:
```shell
cp -rn configs_template configs
```
You may edit the files under config directory as you like.
The parts that need editing are marked with a FIXME comment.

## Training and evaluation
You can train and test the models in the following command:
```shell
poetry run python scripts/train.py devices=[0,1]
```

If you only want to do evaluation after training, please use the following command:
```shell
poetry run python scripts/evaluate.py devices=[0] checkpoint_path="/path/to/checkpoint"
```

## Debugging
You can do debugging on local and server environments:

Local environment (using CPU):
```shell
poetry run python src/train.py -cn debug.local.yaml
```
Server environment (using GPU):
```shell
poetry run python src/train.py -cn debug.yaml devices=[0]
```

## Unit tests

```shell
poetry run pytest
```
