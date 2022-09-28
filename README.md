# KWJA: Kyoto-Waseda Japanese Analyzer

[![test](https://github.com/ku-nlp/kwja/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/kwja/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/kwja/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/kwja)

KWJA is a Japanese language analyzer based on pre-trained language models.
KWJA performs many language analysis tasks, including:
- Typo correction
- Tokenization
- Morphological analysis
- Named entity recognition
- Dependency parsing
- PAS analysis
- Coreference resolution
- Discourse relation analysis
- etc.

## Requirements

- Python: 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Getting Started

Install KWJA with pip:

```shell
$ pip install kwja
```

Perform language analysis with the `kwja` command (the result is in the KNP format):

```shell
# Analyze a text
$ kwja --text "月が綺麗ですね。死んでもいいわ。"

# Analyze a text file
$ kwja --file path/to/file.txt
```

## Citation

```bibtex
@InProceedings{植田2022,
  author    = {植田 暢大 and 大村 和正 and 児玉 貴志 and 清丸 寛一 and 村脇 有吾 and 河原 大輔 and 黒橋 禎夫},
  title     = {KWJA：汎用言語モデルに基づく日本語解析器},
  booktitle = {第253回自然言語処理研究会},
  year      = {2022},
  address   = {京都},
}
```
