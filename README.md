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

You must prepare config files and a `.env` file:
1. Copy base config file and edit `work_dir`
```shell
cp configs/base_template.yaml configs/base.yaml
```
2. Create config file for the task you want to do:
```shell
touch TASK_NAME.yaml TASK_NAME.debug.yaml
```
Please refer to existing tasks when editing and add your task to `configs/module/default.yaml`

3. Create a `.env` file and set `DATA_DIR` and `MODEL_DIR`.
```shell
echo DATA_DIR="/path/to/data_dir" >> .env
echo MODEL_DIR="/path/to/model_dir" >> .env
```

## Preprocessing
If you want to use the word segmenter, please prepare a word matcher in advance with the following command.
```shell
poetry run python src/jula/preprocessors/wiki_ene_dic.py
  --input-json-path "/path/to/wiki_ene_json_file"
```
Options:
- `--output-dir, -o`: path to directory to save. Default: `./data`
- `--save-filtered-results, -s`: whether to create an intermediate file to save the filtering results.

## Build datasets for training word module
You must have access to KyotoCorpusFull
```shell
./scripts/build_datasets.sh -a /path/to/activate -w ./work_dir -s /path/to/scripts -j 2 -o /path/to/output
```

## Training and evaluation
You can train and test the models in the following command:
```shell
# For training and evaluating word segmenter
poetry run python scripts/train.py -cn char_module devices=[0,1]
```

If you only want to do evaluation after training, please use the following command:
```shell
# For evaluating word segmenter
poetry run python scripts/predict.py -cn char_module devices=[0] checkpoint_path="/path/to/checkpoint"
```

## Debugging
You can do debugging on local and server environments:

Local environment (using CPU):
```shell
# For debugging word segmenter
poetry run python scripts/train.py -cn char_module.debug devices=1
```
Server environment (using GPU):
```shell
# For debugging word segmenter
poetry run python scripts/train.py -cn char_module.debug devices=[0]
```

## Unit tests

```shell
poetry run pytest
```
