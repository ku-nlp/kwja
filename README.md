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
1. Copy base config file and edit `work_dir`
```shell
cp configs/base_template.yaml configs/base.yaml
```
2. Create config file for the task you want to do:
```shell
touch TASK_NAME.yaml TASK_NAME.debug.yaml
```
Please refer to existing tasks when editing.

## Training and evaluation
You can train and test the models in the following command:
```shell
# For training and evaluating word segmentor
poetry run python scripts/train.py -cn word_segmentor.yaml devices=[0,1]
```

If you only want to do evaluation after training, please use the following command:
```shell
# For evaluating word segmentor
poetry run python scripts/evaluate.py -cn word_segmentor.yaml devices=[0] checkpoint_path="/path/to/checkpoint"
```

## Debugging
You can do debugging on local and server environments:

Local environment (using CPU):
```shell
# For debugging word segmentor
poetry run python scripts/train.py -cn word_segmentor.debug devices=1
```
Server environment (using GPU):
```shell
# For debugging word segmentor
poetry run python scripts/train.py -cn word_segmentor.debug devices=[0]
```

## Unit tests

```shell
poetry run pytest
```
