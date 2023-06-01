# KWJA: Kyoto-Waseda Japanese Analyzer[^1]

[^1]: Pronunciation: [/kuʒa/](http://ipa-reader.xyz/?text=%20ku%CA%92a)

[![test](https://github.com/ku-nlp/kwja/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/kwja/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/kwja/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/kwja)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ku-nlp/kwja)](https://www.codefactor.io/repository/github/ku-nlp/kwja)
[![PyPI](https://img.shields.io/pypi/v/kwja)](https://pypi.org/project/kwja/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kwja)

[[Paper]](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=220232&item_no=1&page_id=13&block_id=8)
[[Slides]](https://speakerdeck.com/nobug/kyoto-waseda-japanese-analyzer)

KWJA is a Japanese language analyzer based on pre-trained language models.
KWJA performs many language analysis tasks, including:
- Typo correction
- Word segmentation
- Word normalization
- Morphological analysis
- Word feature tagging
- NER (Named Entity Recognition)
- Base phrase feature tagging
- Dependency parsing
- PAS analysis
- Bridging reference resolution
- Coreference resolution
- Discourse relation analysis

## Requirements

- Python: 3.8+
- Dependencies: See [pyproject.toml](./pyproject.toml).
- GPUs with CUDA 11.7 (optional)

## Getting Started

Install KWJA with pip:

```shell
$ pip install kwja
```

Perform language analysis with the `kwja` command (the result is in the KNP format):

```shell
# Analyze a text
$ kwja --text "KWJAは日本語の統合解析ツールです。汎用言語モデルを利用し、様々な言語解析を統一的な方法で解いています。"

# Analyze text files and write the result to a file
$ kwja --filename path/to/file1.txt --filename path/to/file2.txt > path/to/analyzed.knp

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

`--model-size`: Model size to be used. Please specify 'tiny', 'base' (default) or 'large'.

`--device`: Device to be used. Please specify 'cpu' or 'gpu'.

`--typo-batch-size`: Batch size for typo module. This batch size is also used for the senter module.

`--senter-batch-size`: Batch size for senter module.

`--seq2seq-batch-size`: Batch size for seq2seq module.

`--char-batch-size`: Batch size for char module.

`--word-batch-size`: Batch size for word module.

`--tasks`: Tasks to be performed.
  - `typo`: Typo correction
  - `senter`: Sentence segmentation
  - `seq2seq`: Word segmentation, Word normalization, Reading prediction, lemmatization, and Canonicalization.
  - `char`: Word segmentation and Word normalization
  - `word`: Morphological analysis, Named entity recognition, Word feature tagging, Dependency parsing, PAS analysis, Bridging reference resolution, and Coreference resolution

`--config-file`: Path to a custom configuration file.

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

## Configuration

`kwja` can be configured with a configuration file to set the default options.
 Check [Config file content](#config-file-example) for details.

### Config file location

On non-Windows systems `kwja` follows the
[XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
convention for the location of the configuration file.
The configuration dir `kwja` uses is itself named `kwja`.
In that directory it refers to a file named `config.yaml`.
For most people it should be enough to put their config file at `~/.config/kwja/config.yaml`.
You can also provide a configuration file in a non-standard location with: `kwja --config-file <path>`

### Config file example

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

## Performance Table

- The performance on each task except typo correction and discourse relation analysis is the mean over all the corpora (KC, KWDLC, Fuman, and WAC) and over three runs with different random seeds.
- We set the learning rate of RoBERTa<sub>LARGE</sub> (word) to 2e-5 because we failed to fine-tune it with a higher learning rate.
  Other hyperparameters are the same described in configs, which are tuned for DeBERTa<sub>BASE</sub>.

<table>
  <tr>
    <th rowspan="2" colspan="2">Task</th>
    <th colspan="6">Model</th>
  </tr>
  <tr>
    <th>
        RoBERTa<sub>BASE</sub><br>
        (
            <a href="https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/nlp-waseda/roberta-base-japanese">word</a>
        )
    </th>
    <th>
        DeBERTa<sub>BASE</sub><br>
        (
            <a href="https://huggingface.co/ku-nlp/deberta-v2-base-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/ku-nlp/deberta-v2-base-japanese">word</a>
        )
    </th>
    <th>
        RoBERTa<sub>LARGE</sub><br>
        (
            <a href="https://huggingface.co/ku-nlp/roberta-large-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/nlp-waseda/roberta-large-japanese-seq512">word</a>
        )
    </th>
    <th>
        DeBERTa<sub>LARGE</sub><br>
        (
            <a href="https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/ku-nlp/deberta-v2-large-japanese">word</a>
        )
    </th>
    <th>
        mT5<sub>base</sub><br>
        (<a href="https://huggingface.co/google/mt5-base">seq2seq</a>)
    </th>
    <th>
        mT5<sub>large</sub><br>
        (<a href="https://huggingface.co/google/mt5-large">seq2seq</a>)
    </th>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Typo Correction</th>
    <td>TBU</td>
    <td>TBU</td>
    <td>TBU</td>
    <td>TBU</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Sentence Segmentation</th>
    <td>TBU</td>
    <td>TBU</td>
    <td>TBU</td>
    <td>TBU</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Word Segmentation</th>
    <td>98.5</td>
    <td>98.6</td>
    <td>98.7</td>
    <td style="font-weight: bold;">98.9</td>
    <td>97.7</td>
    <td>98.9</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Word Normalization</th>
    <td>44.0</td>
    <td>39.2</td>
    <td>39.8</td>
    <td style="font-weight: bold;">46.0</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th rowspan="7" style="text-align: center;">Morphological<br>Analysis</th>
    <th style="text-align: left;">POS</th>
    <td>99.3</td>
    <td style="font-weight: bold;">99.4</td>
    <td>99.3</td>
    <td style="font-weight: bold;">99.4</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">sub-POS</th>
    <td>98.1</td>
    <td style="font-weight: bold;">98.4</td>
    <td>98.2</td>
    <td style="font-weight: bold;">98.4</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">conjugation type</th>
    <td>99.4</td>
    <td style="font-weight: bold;">99.5</td>
    <td>99.2</td>
    <td>99.4</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">conjugation form</th>
    <td>99.5</td>
    <td style="font-weight: bold;">99.6</td>
    <td>99.4</td>
    <td style="font-weight: bold;">99.6</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">reading</th>
    <td>95.5</td>
    <td>95.2</td>
    <td>90.8</td>
    <td>95.1</td>
    <td>95.5</td>
    <td style="font-weight: bold;">96.2</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">lemma</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>97.2</td>
    <td style="font-weight: bold;">97.5</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">canon</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>94.7</td>
    <td style="font-weight: bold;">95.5</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Named Entity Recognition</th>
    <td>83.0</td>
    <td>83.8</td>
    <td>82.1</td>
    <td style="font-weight: bold;">84.6</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th rowspan="2" style="text-align: center;">Linguistic<br>Feature<br>Tagging</th>
    <th style="text-align: left;">word</th>
    <td>98.3</td>
    <td style="font-weight: bold;">98.5</td>
    <td style="font-weight: bold;">98.5</td>
    <td>98.4</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th style="text-align: left;">base phrase</th>
    <td>86.6</td>
    <td style="font-weight: bold;">89.5</td>
    <td>86.4</td>
    <td>89.3</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Dependency Parsing</th>
    <td>92.9</td>
    <td>93.4</td>
    <td style="font-weight: bold;">93.8</td>
    <td>93.3</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Pas Analysis</th>
    <td>74.2</td>
    <td>76.7</td>
    <td>75.3</td>
    <td style="font-weight: bold;">76.9</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Bridging Reference Resolution</th>
    <td>66.5</td>
    <td style="font-weight: bold;">67.3</td>
    <td>65.2</td>
    <td>67.0</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Coreference Resolution</th>
    <td>74.9</td>
    <td style="font-weight: bold;">78.4</td>
    <td>75.9</td>
    <td>78.0</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="text-align: right;">
    <th colspan="2" style="text-align: center;">Discourse Relation Analysis</th>
    <td>42.2</td>
    <td style="font-weight: bold;">44.8</td>
    <td>41.3</td>
    <td>41.0</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

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

```bibtex
@InProceedings{児玉2023,
  author    = {児玉 貴志 and 植田 暢大 and 大村 和正 and 清丸 寛一 and 村脇 有吾 and 河原 大輔 and 黒橋 禎夫},
  title     = {テキスト生成モデルによる日本語形態素解析},
  booktitle = {言語処理学会 第29回年次大会},
  year      = {2023},
  address   = {沖縄},
}
```

## Reference

- [KNP format](http://cr.fvcrc.i.nagoya-u.ac.jp/~sasano/knp/format.html)
