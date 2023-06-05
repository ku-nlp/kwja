# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Support executing kwja with `python -m kwja`.

### Fixed
- Fix a bug in senter prediction.
- Fix a bug in interactive mode.

## [v2.1.1] - 2023-06-03
### Fixed
- Fix a bug in the interactive mode.
- Fix a bug in the seq2seq model's output.

## [v2.1.0] - 2023-06-02
### Added
- Support Python 3.11.
- Support NN-based sentence segmentation.
  ```shell
  kwja --tasks senter --text "モーニング娘。は日本のアイドルグループです。"
  ```
- Support multiple files as input.
  ```shell
  kwja --filename file1.txt --filename file2.txt
  ```
- Introduce a config file. You can specify some options in `XDG_CONFIG_HOME/kwja/config.yaml`.
  ```yaml
  model_size: base
  device: cpu
  num_workers: 0
  torch_compile: false
  typo_batch_size: 1
  senter_batch_size: 1
  seq2seq_batch_size: 1
  char_batch_size: 1
  word_batch_size: 1
  ```
- Implement padding truncation of word module to accelerate inference.
- Support Windows.

### Changed
- Support CUDA 11.7 by default instead of CUDA 10.x.
- Skip typo correction by default.
- Optimize package requirements for faster loading.
- Optimize model initialization for faster loading.
- Replace mt5 models with t5 models pre-trained on Japanese corpora in seq2seq module.
- Use partially annotated data for word normalization to train seq2seq module.

### Removed
- Remove the discourse module.

### Fixed
- Fix a bug that warning messages are shown when Juman++ and/or KNP are not installed.
- Fix a bug that document IDs are not assigned properly when a text file is given as input.

## [2.0.0] - 2023-03-14

### Added
- Introduce the seq2seq module for more accurate reading prediction and canonicalization.

### Changed
- Replace RoBERTa-based models with DeBERTaV2-based models.
- Support CUDA 11.7 by default instead of CUDA 10.2.

### Fixed
- Fix many minor bugs.

## [1.4.2] - 2023-02-22

### Fixed
- Fix a bug with analysis results not being output.

## [1.4.1] - 2023-01-25

### Fixed
- Fix a bug where checkpoint is not found

## [1.4.0] - 2023-01-25

### Added
- Add an option for which tasks to be performed.

### Fixed
- Fix a corner case where a long sentence can be deleted when document splitting.

## [1.3.0] - 2023-01-23

### Added
- Enable progress bar while executing kwja command.
- Add benchmark script.
- Implement text normalization in char module.
- Add tiny model.

### Fixed
- Fix scripts for building datasets to support latest rhoknp and remove dependency on kyoto-reader.
- Support versioning of local cache directory.
- Stash unsuitable documents so as not to discard them while applying typo module.
- Fix bugs of document_split_stride, reading aligner, and writers.
- Fix phrase masking in cohesion analysis

### Removed
- Remove unused main dependencies, `python-Levenshtein`, `ipadic`, `tinydb`, `BetterJSONStorage`, and `dartsclone`.

## [1.2.2] - 2022-11-07

### Fixed
- Fix a bug where cohesion analysis results are sometimes weird.

## [1.2.1] - 2022-11-04

### Fixed
- Fix bugs of word module writer and interactive mode.

## [1.2.0] - 2022-10-27

### Added
- Add option to change batch size to CLI.
  - `--typo-batch-size`, `--char-batch-size`, and `--word-batch-size`.
- Add large model.

### Fixed
- Output predictions per batch to avoid out of memory error.
- Allow input files containing multiple documents in one file from command line.
- Use pure-cdb for storing JumanDIC instead of TinyDB.

## [1.1.2] - 2022-10-13

### Fixed
- Fix a bug where the CLI does not work due to a missing dependency.
- Relax the version constraint on `torch`.

## [1.1.1] - 2022-10-12

### Fixed
- Output the instruction message of the CLI tool to stderr instead of stdout.

## [1.1.0] - 2022-10-11

### Added
- Interactive mode.
- Support for Python 3.8.

### Fixed
- Use GPU for analyses if available.

### Removed
- Remove unused dependencies, `wandb` and `kyoto-reader`.

## [1.0.3] - 2022-10-03

### Added
- Add `--version` option to CLI.

### Fixed
- Analyze unnormalized texts in word module after word normalization.

## [1.0.2] - 2022-10-01

### Fixed
- Fix dependency parsing.

## [1.0.1] - 2022-09-28

### Removed
- Remove an unnecessary dependency, `fugashi`.

[Unreleased]: https://github.com/ku-nlp/kwja/compare/v2.1.1...HEAD
[2.1.1]: https://github.com/ku-nlp/kwja/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/ku-nlp/kwja/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/ku-nlp/kwja/compare/v1.4.2...v2.0.0
[1.4.2]: https://github.com/ku-nlp/kwja/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/ku-nlp/kwja/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/ku-nlp/kwja/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/ku-nlp/kwja/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/ku-nlp/kwja/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/ku-nlp/kwja/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/ku-nlp/kwja/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/ku-nlp/kwja/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/ku-nlp/kwja/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/ku-nlp/kwja/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/ku-nlp/kwja/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/ku-nlp/kwja/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/ku-nlp/kwja/compare/v1.0.0...v1.0.1
