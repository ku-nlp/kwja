# KWJA: Kyoto-Waseda Japanese Analyzer

[![test](https://github.com/ku-nlp/kwja/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/kwja/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/kwja/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/kwja)
[![PyPI](https://img.shields.io/pypi/v/kwja)](https://pypi.org/project/kwja/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kwja)

[[Paper]](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=220232&item_no=1&page_id=13&block_id=8)
[[Slides]](https://speakerdeck.com/nobug/kyoto-waseda-japanese-analyzer)

KWJA is a Japanese language analyzer based on pre-trained language models.
KWJA performs many language analysis tasks, including:
- Typo correction
- Tokenization
- Word normalization
- Morphological analysis
- Named entity recognition
- Word feature tagging
- Dependency parsing
- PAS analysis
- Bridging reference resolution
- Coreference resolution
- Discourse relation analysis

## Requirements

- Python: 3.8+
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Getting Started

Install KWJA with pip:

```shell
$ pip install kwja
```

Perform language analysis with the `kwja` command (the result is in the KNP format):

```shell
# Analyze a text
$ kwja --text "KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。"

# Analyze a text file and write the result to a file
$ kwja --file path/to/file.txt > path/to/analyzed.knp

# Analyze texts interactively
$ kwja
Please end your input with a new line and type "EOD"
KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。
EOD
```

The output is in the KNP format, like the following:

```
# S-ID:202210010000-0-0 kwja:1.0.2
* 2D
+ 5D <rel type="=" target="ツール" sid="202210011918-0-0" id="5"/><体言><NE:ARTIFACT:KWJA>
KWJA ＫWＪＡ KWJA 名詞 6 固有名詞 3 * 0 * 0 <基本句-主辞>
は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は" <代表表記:は/は>
* 2D
+ 2D <体言>
日本 にほん 日本 名詞 6 地名 4 * 0 * 0 "代表表記:日本/にほん 地名:国" <代表表記:日本/にほん><地名:国><基本句-主辞>
+ 4D <体言><係:ノ格>
語 ご 語 名詞 6 普通名詞 1 * 0 * 0 "代表表記:語/ご 漢字読み:音 カテゴリ:抽象物" <代表表記:語/ご><漢字読み:音><カテゴリ:抽象物><基本句-主辞>
の の の 助詞 9 接続助詞 3 * 0 * 0 "代表表記:の/の" <代表表記:の/の>
...
```

Here are some other options for `kwja` command:

`--model-size`: Model size to be used. Please specify 'base' or 'large'.

`--device`: Device to be used. Please specify 'cpu' or 'gpu'.

`--typo-batch-size`: Batch size for typo module.

`--char-batch-size`: Batch size for char module.

`--word-batch-size`: Batch size for word module.

`--discourse`: Whether to perform discourse relation analysis. Default value is True. If you do not need the results of discourse relation analysis, please specify `--no-discourse`.


You can read a KNP format file with [rhoknp](https://github.com/ku-nlp/rhoknp).

```python
from rhoknp import Document
with open("analyzed.knp") as f:
    parsed_document = Document.from_knp(f.read())
```

For more details about KNP format, see [Reference](#reference).

## Usage from Python

Make sure you have `kwja` command in your path:

```shell
$ which kwja
/path/to/kwja
```

Install [rhoknp](https://github.com/ku-nlp/rhoknp):

```shell
$ pip install rhoknp
```

Perform language analysis with the `kwja` instance:

```python
from rhoknp import KWJA
kwja = KWJA()
analyzed_document = kwja.apply(
    "KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。"
)
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

## Reference

- [KNP format](http://cr.fvcrc.i.nagoya-u.ac.jp/~sasano/knp/format.html)
