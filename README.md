# KWJA: Kyoto-Waseda Japanese Analyzer[^1]

[^1]: Pronunciation: [/kuʒa/](http://ipa-reader.xyz/?text=%20ku%CA%92a)

[![test](https://github.com/ku-nlp/kwja/actions/workflows/test.yml/badge.svg)](https://github.com/ku-nlp/kwja/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ku-nlp/kwja/branch/main/graph/badge.svg?token=A9FWWPLITO)](https://codecov.io/gh/ku-nlp/kwja)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ku-nlp/kwja)](https://www.codefactor.io/repository/github/ku-nlp/kwja)
[![PyPI](https://img.shields.io/pypi/v/kwja)](https://pypi.org/project/kwja/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kwja)

[[Paper]](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=220232&item_no=1&page_id=13&block_id=8)
[[Slides]](https://speakerdeck.com/nobug/kyoto-waseda-japanese-analyzer)

KWJA is an integrated Japanese text analyzer based on foundation models.
KWJA performs many text analysis tasks, including:
- Typo correction
- Sentence segmentation
- Word segmentation
- Word normalization
- Morphological analysis
- Word feature tagging
- Base phrase feature tagging
- NER (Named Entity Recognition)
- Dependency parsing
- Predicate-argument structure (PAS) analysis
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

If you use Windows and PowerShell, you need to set `PYTHONUTF8` environment variable to `1`:

```shell
> $env:PYTHONUTF8 = "1"
> kwja ...
````

The output is in the KNP format, which looks like the following:

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

Here are options for `kwja` command:

- `--text`: Text to be analyzed.

- `--filename`: Path to a text file to be analyzed. You can specify this option multiple times.

- `--model-size`: Model size to be used. Specify one of `tiny`, `base` (default), and `large`.

- `--device`: Device to be used. Specify `cpu` or `gpu`. If not specified, the device is automatically selected.

- `--typo-batch-size`: Batch size for typo module.

- `--senter-batch-size`: Batch size for senter module.

- `--seq2seq-batch-size`: Batch size for seq2seq module.

- `--char-batch-size`: Batch size for character module.

- `--word-batch-size`: Batch size for word module.

- `--tasks`: Tasks to be performed. Specify one or more of the following values separated by commas:
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
You can also provide a configuration file in a non-standard location with an environment variable `KWJA_CONFIG_FILE` or a command line option `--config-file`.

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

- typo, senter, character, and word modules
  - The performance on each task except typo correction and discourse relation analysis is the mean over all the corpora (KC, KWDLC, Fuman, and WAC) and over three runs with different random seeds.
  - We set the learning rate of RoBERTa<sub>LARGE</sub> (word) to 2e-5 because we failed to fine-tune it with a higher learning rate.
    Other hyperparameters are the same described in configs, which are tuned for DeBERTa<sub>BASE</sub>.
- seq2seq module
  - The performance on each task is the mean over all the corpora (KC, KWDLC, Fuman, and WAC).
    - \* denotes results of a single run
  - Scores are calculated using a separate [script](https://github.com/ku-nlp/kwja/blob/main/scripts/view_seq2seq_results.py) from the character and word modules.

<table>
  <thead>
    <tr>
      <th rowspan="2" colspan="2">Task</th>
      <th colspan="6">Model</th>
    </tr>
    <tr>
      <th>
        v1.x base<br>
        (
            <a href="https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/nlp-waseda/roberta-base-japanese">word</a>
        )
      </th>
      <th>
        v2.x base<br>
        (
            <a href="https://huggingface.co/ku-nlp/deberta-v2-base-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/ku-nlp/deberta-v2-base-japanese">word</a> /
            <a href="https://huggingface.co/retrieva-jp/t5-base-long">seq2seq</a>
        )
      </th>
      <th>
        v1.x large<br>
        (
            <a href="https://huggingface.co/ku-nlp/roberta-large-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/nlp-waseda/roberta-large-japanese-seq512">word</a>
        )
      </th>
      <th>
        v2.x large<br>
        (
            <a href="https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm">char</a>,
            <a href="https://huggingface.co/ku-nlp/deberta-v2-large-japanese">word</a> /
            <a href="https://huggingface.co/retrieva-jp/t5-large-long">seq2seq</a>
        )
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="2">Typo Correction</th>
      <td>79.0</td>
      <td>76.7</td>
      <td>80.8</td>
      <td>83.1</td>
    </tr>
    <tr>
      <th colspan="2">Sentence Segmentation</th>
      <td>-</td>
      <td>98.4</td>
      <td>-</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th colspan="2">Word Segmentation</th>
      <td>98.5</td>
      <td>98.1 / 98.2*</td>
      <td>98.7</td>
      <td>98.4 / 98.4*</td>
    </tr>
    <tr>
      <th colspan="2">Word Normalization</th>
      <td>44.0</td>
      <td>15.3</td>
      <td>39.8</td>
      <td>48.6</td>
    </tr>
    <tr>
      <th rowspan="7">Morphological Analysis</th>
      <th>POS</th>
      <td>99.3</td>
      <td>99.4</td>
      <td>99.3</td>
      <td>99.4</td>
    </tr>
    <tr>
      <th>sub-POS</th>
      <td>98.1</td>
      <td>98.5</td>
      <td>98.2</td>
      <td>98.5</td>
    </tr>
    <tr>
      <th>conjtype</th>
      <td>99.4</td>
      <td>99.6</td>
      <td>99.2</td>
      <td>99.6</td>
    </tr>
    <tr>
      <th>conjform</th>
      <td>99.5</td>
      <td>99.7</td>
      <td>99.4</td>
      <td>99.7</td>
    </tr>
    <tr>
      <th>reading</th>
      <td>95.5</td>
      <td>95.4 / 96.2*</td>
      <td>90.8</td>
      <td>95.6 / 96.8*</td>
    </tr>
    <tr>
      <th>lemma</th>
      <td>-</td>
      <td>- / 97.8*</td>
      <td>-</td>
      <td>- / 98.1*</td>
    </tr>
    <tr>
      <th>canon</th>
      <td>-</td>
      <td>- / 95.2*</td>
      <td>-</td>
      <td>- / 95.9*</td>
    </tr>
    <tr>
      <th colspan="2">Named Entity Recognition</th>
      <td>83.0</td>
      <td>84.6</td>
      <td>82.1</td>
      <td>85.9</td>
    </tr>
    <tr>
      <th rowspan="2">Linguistic Feature Tagging</th>
      <th>word</th>
      <td>98.3</td>
      <td>98.6</td>
      <td>98.5</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>base phrase</th>
      <td>86.6</td>
      <td>93.6</td>
      <td>86.4</td>
      <td>93.4</td>
    </tr>
    <tr>
      <th colspan="2">Dependency Parsing</th>
      <td>92.9</td>
      <td>93.5</td>
      <td>93.8</td>
      <td>93.6</td>
    </tr>
    <tr>
      <th colspan="2">Pas Analysis</th>
      <td>74.2</td>
      <td>76.9</td>
      <td>75.3</td>
      <td>77.5</td>
    </tr>
    <tr>
      <th colspan="2">Bridging Reference Resolution</th>
      <td>66.5</td>
      <td>67.3</td>
      <td>65.2</td>
      <td>67.5</td>
    </tr>
    <tr>
      <th colspan="2">Coreference Resolution</th>
      <td>74.9</td>
      <td>78.6</td>
      <td>75.9</td>
      <td>79.2</td>
    </tr>
    <tr>
      <th colspan="2">Discourse Relation Analysis</th>
      <td>42.2</td>
      <td>39.2</td>
      <td>41.3</td>
      <td>44.3</td>
    </tr>
  </tbody>
</table>

## Citation

```bibtex
@InProceedings{Ueda2023a,
  author    = {Nobuhiro Ueda and Kazumasa Omura and Takashi Kodama and Hirokazu Kiyomaru and Yugo Murawaki and Daisuke Kawahara and Sadao Kurohashi},
  title     = {KWJA: A Unified Japanese Analyzer Based on Foundation Models},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year      = {2023},
  address   = {Toronto, Canada},
}
```

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

## License

This software is released under the MIT License, see [LICENSE](LICENSE).

## Reference

- [KNP format](http://cr.fvcrc.i.nagoya-u.ac.jp/~sasano/knp/format.html)
