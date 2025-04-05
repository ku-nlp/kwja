import json
import logging
import random
import re
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

from Levenshtein import opcodes

from kwja.utils.normalization import normalize_text

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", level=logging.DEBUG)


@dataclass(frozen=True)
class Component:
    pre_str: str
    post_str: str
    type: str


class OpType(Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"


def normalize_example(example: dict) -> None:
    example["pre_text"] = normalize_text(example["pre_text"])
    example["post_text"] = normalize_text(example["post_text"])
    for diff in example["diffs"]:
        diff["pre_str"] = normalize_text(diff["pre_str"])
        diff["post_str"] = normalize_text(diff["post_str"])


def decompose(pre_text: str, post_text: str, diffs: List[dict]) -> Optional[List[Component]]:
    # decompose texts into components
    # return None if alignment fails
    components: List[Component] = []
    monitor: Dict[str, str] = {"pre_text": pre_text, "post_text": post_text}
    for diff in diffs:
        if len(diff["pre_str"]) == 0:
            src, tgt = "post", "pre"
            type_ = "insert"
        else:
            src, tgt = "pre", "post"
            type_ = "delete" if len(diff["post_str"]) == 0 else "replace"
        for mo in re.finditer(re.escape(diff[f"{src}_str"]), monitor[f"{src}_text"]):
            start = mo.start()

            src_same = monitor[f"{src}_text"][:start]
            tgt_same = monitor[f"{tgt}_text"][:start]
            if src_same != tgt_same:  # diffs以外にもtypoを含んでいる
                logger.warning(f'"{monitor["pre_text"]}" != "{monitor["post_text"]}" ... skip')
                return None
            src_diff = monitor[f"{src}_text"][start : start + len(diff[f"{src}_str"])]
            tgt_diff = monitor[f"{tgt}_text"][start : start + len(diff[f"{tgt}_str"])]
            if tgt_diff != diff[f"{tgt}_str"]:  # diff[f"{src}_str"]がsrc_textのtypoじゃないところにマッチしている
                continue
            src_cont = monitor[f"{src}_text"][start + len(diff[f"{src}_str"]) : start + len(diff[f"{src}_str"]) + 1]
            tgt_cont = monitor[f"{tgt}_text"][start + len(diff[f"{tgt}_str"]) : start + len(diff[f"{tgt}_str"]) + 1]
            if src_cont != tgt_cont:  # diffの続きが一致しない = typoじゃないところにマッチしている
                continue

            if len(diff["pre_str"]) == 0:
                components.append(Component(pre_str=tgt_same, post_str=src_same, type="equal"))
                components.append(Component(pre_str=tgt_diff, post_str=src_diff, type=type_))
            else:
                components.append(Component(pre_str=src_same, post_str=tgt_same, type="equal"))
                components.append(Component(pre_str=src_diff, post_str=tgt_diff, type=type_))
            monitor.update(
                {
                    "pre_text": monitor["pre_text"][start + len(diff["pre_str"]) :],
                    "post_text": monitor["post_text"][start + len(diff["post_str"]) :],
                }
            )
            break
    if len(monitor["pre_text"]) > 0 and len(monitor["post_text"]) > 0:
        if monitor["pre_text"] != monitor["post_text"]:  # diffs以外にもtypoを含んでいる
            logger.warning(f'"{monitor["pre_text"]}" != "{monitor["post_text"]}" ... skip')
            return None
        components.append(Component(pre_str=monitor["pre_text"], post_str=monitor["post_text"], type="equal"))
    return components


def convert_components_into_tags(components: List[Component], length: int) -> Tuple[List[str], List[str]]:
    # Keep ("K"), Delete ("D"), Replace ("R:x")
    kdr_tags: List[str] = ["K"] * length
    # Insert ("I:x"), Nothing ("_")
    ins_tags: List[str] = ["_"] * length

    cursor = 0
    for component in components:
        if component.type == "equal":  # keep all
            cursor += len(component.pre_str)
        else:
            if component.type == "delete":
                for i in range(len(component.pre_str)):
                    kdr_tags[cursor + i] = "D"
            elif component.type == "insert":
                assert ins_tags[cursor] == "_"
                ins_tags[cursor] = f"I:{component.post_str}"
            elif component.type == "replace":
                for op, i1, i2, j1, j2 in opcodes(component.pre_str, component.post_str):
                    if OpType(op) == OpType.EQUAL:
                        pass
                    elif OpType(op) == OpType.DELETE:
                        for i in range(i1, i2):
                            kdr_tags[cursor + i] = "D"
                    elif OpType(op) == OpType.INSERT:
                        ins_tags[cursor + i1] = f"I:{component.post_str[j1:j2]}"
                    elif OpType(op) == OpType.REPLACE:
                        if i2 - i1 == j2 - j1:
                            for i in range(i2 - i1):
                                kdr_tags[cursor + i1 + i] = f"R:{component.post_str[j1 + i]}"
                        else:
                            logger.warning(f"replace\t{component.pre_str[i1:i2]}\t{component.post_str[j1:j2]}\n")
                    else:
                        raise ValueError("invalid operation")
            else:
                raise ValueError("invalid operation")
            cursor += len(component.pre_str)
    return kdr_tags, ins_tags


def load_examples(in_dir: Path, split: str) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    category2examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    other_examples: List[Dict[str, str]] = []
    with (in_dir / f"{split}.jsonl").open() as f:
        for line in f:
            example: dict = json.loads(line)
            normalize_example(example)
            diffs = [diff for diff in example["diffs"] if diff["category"] != "not-typo"]
            assert len(diffs) > 0
            components: Optional[List[Component]] = decompose(example["pre_text"], example["post_text"], diffs)
            if components is None:
                continue
            kdr_tags, ins_tags = convert_components_into_tags(components, len(example["pre_text"]) + 1)
            example = {
                "pre_text": example["pre_text"],
                "post_text": example["post_text"],
                "kdr_tags": kdr_tags,
                "ins_tags": ins_tags,
            }
            if split == "train" and len(diffs) == 1:
                category2examples[diffs[0]["category"]].append(example)
            else:
                other_examples.append(example)
    return category2examples, other_examples


def save_examples(
    category2examples: Dict[str, List[Dict[str, str]]],
    other_examples: List[Dict[str, str]],
    out_dir: Path,
    split: str,
    num_valid_examples_per_category: int,
) -> None:
    if split == "train":
        train_examples: List[Dict[str, str]] = other_examples
        valid_examples: List[Dict[str, str]] = []
        for category, examples in category2examples.items():
            train_examples.extend(examples[num_valid_examples_per_category:])
            valid_examples.extend(examples[:num_valid_examples_per_category])

        random.shuffle(train_examples)
        train_dir: Path = out_dir / split
        train_dir.mkdir(parents=True, exist_ok=True)
        with (train_dir / f"{split}.jsonl").open(mode="w") as f:
            f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in train_examples) + "\n")

        valid_dir: Path = out_dir / "valid"
        valid_dir.mkdir(parents=True, exist_ok=True)
        with (valid_dir / "valid.jsonl").open(mode="w") as f:
            f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in valid_examples) + "\n")
    elif split == "test":
        test_dir: Path = out_dir / split
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / f"{split}.jsonl").open(mode="w") as f:
            f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in other_examples) + "\n")
    else:
        raise ValueError("invalid split")


def build_multi_char_vocab(out_dir: Path) -> None:
    multi_char_vocab: List[str] = []
    with (out_dir / "train" / "train.jsonl").open() as f:
        for line in f:
            train_example: dict = json.loads(line)
            for ins_tag in train_example["ins_tags"]:
                if ins_tag == "_":
                    continue
                ins_char: str = ins_tag.removeprefix("I:")
                if len(ins_char) > 1:
                    multi_char_vocab.append(ins_char)

    multi_char_vocab_counter = Counter(multi_char_vocab)

    with (out_dir / "multi_char_vocab.txt").open(mode="w") as f:
        for vocab, count in multi_char_vocab_counter.most_common():
            if count > 1:
                f.write(f"{vocab}\n")


def main():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--in-dir",
        type=Path,
        help="path to input directory",
    )
    parser.add_argument(
        "-n",
        "--num-valid-examples-per-category",
        type=int,
        default=1000,
        help=dedent(
            """\
            number of validation examples. \
            The validation data is extracted from the train.jsonl by the number of num-valid-examples-per-category for \
            each of the 8 error categories. \
            Therefore, the actual number of validation data is 8 times the number specified here. \
            The training data is the rest.\
            """
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        help="path to output directory",
    )
    args = parser.parse_args()

    random.seed(0)

    for split in ["train", "test"]:
        category2examples, other_examples = load_examples(args.in_dir, split)
        save_examples(category2examples, other_examples, args.out_dir, split, args.num_valid_examples_per_category)
    build_multi_char_vocab(args.out_dir)


if __name__ == "__main__":
    main()
