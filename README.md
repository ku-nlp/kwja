# KWJA: Kyoto-Waseda Japanese Analyzer

[![test](https://github.com/ku-nlp/kwja/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/kwja/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/kwja/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/kwja)

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

3. Create a `.env` file and set `DATA_DIR`.
```shell
echo DATA_DIR="/path/to/data_dir" >> .env
```

## Preprocessing
If you want to use the word segmenter, please prepare a word matcher in advance with the following command.
```shell
poetry run python src/kwja/preprocessors/wiki_ene_dic.py
  --input-json-path "/path/to/wiki_ene_json_file"
```
Options:
- `--output-dir, -o`: path to directory to save. Default: `./data`
- `--save-filtered-results, -s`: whether to create an intermediate file to save the filtering results.

For morphological analysis, you need to convert JumanDIC in advance with the following commands.
```shell
cd /path/to/JumanDIC
git checkout kwja
make kwja
```
and
```shell
poetry run python src/kwja/preprocessors/preprocess_jumandic.py
  --input-dir /path/to/JumanDIC
  --output-dir /path/to/dic_dir
```
Options:
- `--input-dir, -i`: path to the JumanDIC dir.
- `--output-dir, -o`: path to a directory where processed data are saved.

## Build dataset for training typo module
You must preprocess Japanese Wikipedia Typo Dataset.
```shell
poetry run python src/kwja/preprocessors/preprocess_typo.py
  --input-dir "/path/to/unzipped_typo_dataset_dir"
```
Options:
- `--output-dir, -o`: path to directory to save. Default: `./data`
- `--num-valid-samples, -n`: number of validation data. Default: `1000`

## Build datasets for training word module
"build_datasets.sh" performs formatting KWDLC and annotated FKC corpus.
```shell
./scripts/build_datasets.sh
  -a $(poetry run echo $VIRTUAL_ENV)/bin/activate
  -w /path/to/work_dir
  -s $(realpath ./scripts)
  -j 2
  -o /path/to/output_dir
```
Options:
- `-a`: path to activator
- `-w`: path to working directory
- `-s`: path to scripts
- `-j`: number of jobs
- `-o`: path to output directory

NOTE:
To train word module on Kyoto University Text Corpus, you must have access to it and IREX CRL named entity data.
If you have both access, you can format the corpus with the following commands.
(You may need preprocessing to format IREX CRL named entity data.)
```shell
poetry run python scripts/add_features_to_raw_corpus.py
  KyotoCorpus/knp
  kyoto/knp
  --ne-tags IREX_CRL_NE_data.jmn
  -j 2
poetry run kyoto idsplit \
  --corpus-dir kyoto/knp \
  --output-dir kyoto \
  --train KyotoCorpus/id/full/train.id \
  --valid KyotoCorpus/id/full/dev.id \
  --test KyotoCorpus/id/full/test.id
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
