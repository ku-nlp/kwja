# Developer Guide

## Installation

To prepare the Python environment:

```shell
poetry install
```

You need prepare config files and the `.env` file:

1. Copy base config file and edit `work_dir`

```shell
cp configs/base_template.yaml configs/base.yaml
```

2. Create a `.env` file and set `DATA_DIR`.

```shell
echo DATA_DIR="/path/to/data_dir" >> .env
```

## Preprocessing

For morphological analysis, you need to convert JumanDIC in advance with the following commands.

```shell
cd /path/to/JumanDIC
git checkout kwja
make kwja
```

and

```shell
poetry run python scripts/preprocessors/preprocess_jumandic.py
  --input-dir /path/to/JumanDIC
  --output-dir /path/to/dic_dir
```

Options:

- `--input-dir, -i`: path to the JumanDIC dir.
- `--output-dir, -o`: path to a directory where processed data are saved.

## Building dataset for training typo module

You must preprocess Japanese Wikipedia Typo Dataset.

```shell
poetry run python scripts/preprocessors/preprocess_typo.py
  --input-dir "/path/to/unzipped_typo_dataset_dir"
```

Options:

- `--output-dir, -o`: path to directory to save. Default: `./data`
- `--num-valid-samples, -n`: number of validation data. Default: `1000`

## Building datasets for training word module

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
poetry run python scripts/test.py module=char checkpoint_path="/path/to/checkpoint" devices=[0]
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

## Running unit test

```shell
poetry run pytest
```

## Releasing a new version

- Checkout `main` branch
- Make sure the new version is supported in `_get_model_version` function in `src/kwja/cli/utils.py`
- Update `CHANGELOG.md`
- Edit `pyproject.toml` to update `tool.poetry.version`
- Update dependencies

    ```shell
    poetry update
    ```

- Add a new tag and push changes

    ```shell
    git tag -a v0.1.0 -m "Release v0.1.0"
    git push --follow-tags
    ```

- If CI is passed, publish to PyPI

    ```shell
    poetry build
    poetry publish [--username $PYPI_USERNAME] [--password $PYPI_PASSWORD]
    ```
