# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
[1.1.0]: https://github.com/ku-nlp/kwja/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/ku-nlp/kwja/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/ku-nlp/kwja/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/ku-nlp/kwja/compare/v1.0.0...v1.0.1