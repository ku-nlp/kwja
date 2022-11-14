# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Fix scripts for building datasets to support latest rhoknp and remove dependency on kyoto-reader.
- Support versioning of local cache directory.

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

[Unreleased]: https://github.com/ku-nlp/kwja/compare/v1.1.0...HEAD
[1.2.0]: https://github.com/ku-nlp/kwja/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/ku-nlp/kwja/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/ku-nlp/kwja/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/ku-nlp/kwja/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/ku-nlp/kwja/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/ku-nlp/kwja/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/ku-nlp/kwja/compare/v1.0.0...v1.0.1
