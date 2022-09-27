import json
import logging
import random
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from Levenshtein import opcodes

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", level=logging.DEBUG)


@dataclass(frozen=True)
class TypoDiff:
    pre_str: str
    post_str: str


@dataclass(frozen=True)
class Block:
    text: str
    diff: Optional[TypoDiff]


class OpcodeType(Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"


class TypoPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def decompose(pre_text: str, post_text: str, typo_diffs: list[TypoDiff]) -> Optional[list[Block]]:
        # decompose pre_text into blocks
        # return None if alignment fails
        assert len(typo_diffs) > 0

        s_idx: int = 0
        remaining_pre_text: str = pre_text
        remaining_post_text: str = post_text
        blocks: list[Block] = []
        for diff in typo_diffs:
            if len(diff.pre_str) > 0 and len(diff.post_str) > 0:
                # both pre_str and post_str are non-null: check if post_text contains post_str at the position s_idx
                for m in re.finditer(re.escape(diff.pre_str), remaining_pre_text):
                    s_idx = m.start()
                    if remaining_post_text[s_idx : s_idx + len(diff.post_str)] == diff.post_str:
                        if s_idx + len(diff.pre_str) < len(remaining_pre_text):
                            pre_idx = s_idx + len(diff.pre_str)
                            post_idx = s_idx + len(diff.post_str)
                            if (
                                remaining_pre_text[pre_idx : pre_idx + 1]
                                != remaining_post_text[post_idx : post_idx + 1]
                            ):
                                continue
                        if remaining_pre_text[:s_idx] != remaining_post_text[:s_idx]:
                            logger.warning("remaining_pre_text != remaining_post_text...skip")
                            return None
                        blocks.append(Block(text=remaining_pre_text[:s_idx], diff=None))
                        blocks.append(Block(text="", diff=TypoDiff(pre_str=diff.pre_str, post_str=diff.post_str)))
                        remaining_pre_text = remaining_pre_text[s_idx + len(diff.pre_str) :]
                        remaining_post_text = remaining_post_text[s_idx + len(diff.post_str) :]
                        break
                else:
                    logger.warning(f"pre_str {diff.pre_str} not found...skip")
                    return None
            elif len(diff.pre_str) > 0:
                # pre_str is non-null and post_str is null: check if the preceding sentences + the next chars (if any) are the same
                for m in re.finditer(re.escape(diff.pre_str), remaining_pre_text):
                    s_idx = m.start()
                    pre_fragment: str = remaining_pre_text[:s_idx]
                    post_fragment: str = remaining_post_text[: len(pre_fragment)]
                    if s_idx + len(diff.pre_str) < len(remaining_pre_text):
                        post_fragment += remaining_post_text[len(pre_fragment) : len(pre_fragment) + 1]
                        pre_idx = s_idx + len(diff.pre_str)
                        pre_fragment += remaining_pre_text[pre_idx : pre_idx + 1]
                    if pre_fragment == post_fragment:
                        blocks.append(Block(text=remaining_pre_text[:s_idx], diff=None))
                        blocks.append(Block(text="", diff=TypoDiff(pre_str=diff.pre_str, post_str=diff.post_str)))
                        remaining_pre_text = remaining_pre_text[s_idx + len(diff.pre_str) :]
                        remaining_post_text = remaining_post_text[s_idx:]
                        break
                else:
                    logger.warning(f"pre_str {diff.pre_str} not found...skip")
                    return None
            elif len(diff.post_str) > 0:
                for m in re.finditer(re.escape(diff.post_str), remaining_post_text):
                    s_idx = m.start()
                    post_fragment = remaining_post_text[:s_idx] + diff.pre_str
                    pre_fragment = remaining_pre_text[: len(post_fragment)]
                    if s_idx + len(diff.post_str) < len(remaining_post_text):
                        pre_fragment += remaining_pre_text[len(post_fragment) : len(post_fragment) + 1]
                        post_pos = s_idx + len(diff.post_str)
                        post_fragment += remaining_post_text[post_pos : post_pos + 1]
                    if pre_fragment == post_fragment:
                        blocks.append(Block(text=remaining_post_text[:s_idx], diff=None))
                        blocks.append(Block(text="", diff=TypoDiff(pre_str=diff.pre_str, post_str=diff.post_str)))
                        remaining_pre_text = remaining_pre_text[s_idx:]
                        remaining_post_text = remaining_post_text[s_idx + len(diff.post_str) :]
                        break
                else:
                    logger.warning(f"post_str {diff.post_str} not found...skip")
                    return None
            else:
                logger.warning("broken diff...skip")
                return None

        if len(remaining_pre_text) > 0:
            if remaining_pre_text != remaining_post_text:
                logger.warning("remaining_pre_text != remaining_post_text...skip")
                return None
            blocks.append(Block(text=remaining_pre_text[:s_idx], diff=None))
        return blocks

    @staticmethod
    def generate_opn(pre_text: str, blocks: list[Block]) -> Tuple[list[str], list[str]]:
        # Keep ("K"), Delete ("D"), Replace ("R:x")
        kdr_opns: list[str] = ["K"] * (len(pre_text) + 1)
        # Insert ("I:x"), Nothing ("_")
        ins_opns: list[str] = ["_"] * (len(pre_text) + 1)

        s_idx = 0
        for block in blocks:
            if block.diff is None:  # keep all
                s_idx += len(block.text)
            else:
                diff = block.diff
                if len(diff.pre_str) == 0:  # insert post_str
                    assert ins_opns[s_idx] == "_"
                    ins_opns[s_idx] = f"I:{diff.post_str}"
                elif len(diff.post_str) == 0:  # delete pre_str
                    for i in range(len(diff.pre_str)):
                        kdr_opns[s_idx + i] = "D"
                    s_idx += len(diff.pre_str)
                else:
                    for opn, i1, i2, j1, j2 in opcodes(diff.pre_str, diff.post_str):
                        if OpcodeType(opn) == OpcodeType.EQUAL:
                            pass
                        elif OpcodeType(opn) == OpcodeType.DELETE:
                            for i in range(i1, i2):
                                kdr_opns[s_idx + i] = "D"
                        elif OpcodeType(opn) == OpcodeType.INSERT:
                            ins_opns[s_idx + i1] = f"I:{diff.post_str[j1:j2]}"
                        elif OpcodeType(opn) == OpcodeType.REPLACE:
                            if i2 - i1 == j2 - j1:
                                for i in range(i2 - i1):
                                    kdr_opns[s_idx + i1 + i] = f"R:{diff.post_str[j1 + i]}"
                            else:
                                logger.warning(f"replace\t{diff.pre_str[i1:i2]}\t{diff.post_str[j1:j2]}\n")
                        else:
                            raise ValueError("unsupported operation!")
                    s_idx += len(diff.pre_str)
        return kdr_opns, ins_opns

    @staticmethod
    def make_multi_char_vocab(train_path: Path, output_dir: Path) -> None:
        multi_char_vocabs: list[str] = []
        with train_path.open(mode="r") as fr:
            for line in fr:
                train_example: dict = json.loads(line)
                for ins_opn in train_example["inss"]:
                    if ins_opn == "_":
                        continue
                    ins_vocab: str = ins_opn.removeprefix("I:")
                    if len(ins_vocab) > 1:
                        multi_char_vocabs.append(ins_vocab)

        multi_char_vocabs_counter = Counter(multi_char_vocabs)

        with output_dir.joinpath("multi_char_vocab.txt").open("w") as fw:
            for vocab, count in multi_char_vocabs_counter.most_common():
                if count > 1:
                    fw.write(f"{vocab}\n")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="path to input directory",
    )
    parser.add_argument(
        "-n",
        "--num-valid-samples-per-category",
        type=int,
        default=1000,
        help="number of validation data. The evaluation data is extracted from the train.jsonl "
        "by the number of num-valid-samples-per-category for each of the 8 error categories."
        "Therefore, the actual number of evaluation data is 8 times the number specified here."
        "The training data is the rest.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data",
        help="path to directory to save",
    )
    args = parser.parse_args()

    preprocessor: TypoPreprocessor = TypoPreprocessor()
    for filename in ["test", "train"]:
        category2obj = defaultdict(list[str])
        other_objs: list[str] = []
        with Path(f"{args.input_dir}/{filename}.jsonl").open("r") as f:
            for line in f:
                example: dict = json.loads(line)
                diffs = [diff for diff in example["diffs"] if diff["category"] != "not-typo"]
                blocks: Optional[list[Block]] = preprocessor.decompose(
                    pre_text=example["pre_text"],
                    post_text=example["post_text"],
                    typo_diffs=[TypoDiff(pre_str=diff["pre_str"], post_str=diff["post_str"]) for diff in diffs],
                )
                if blocks is None:
                    continue
                kdr_opns, ins_opns = preprocessor.generate_opn(pre_text=example["pre_text"], blocks=blocks)
                obj = {
                    "pre_text": example["pre_text"],
                    "post_text": example["post_text"],
                    "kdrs": kdr_opns,
                    "inss": ins_opns,
                }
                if len(diffs) == 1 and filename == "train":
                    category2obj[diffs[0]["category"]].append(json.dumps(obj, ensure_ascii=False))
                else:
                    other_objs.append(json.dumps(obj, ensure_ascii=False))

        if filename == "test":
            test_dir: Path = Path(f"{args.output_dir}/{filename}")
            test_dir.mkdir(exist_ok=True)
            with test_dir.joinpath(f"{filename}.jsonl").open("w") as f:
                f.write("\n".join(other_objs) + "\n")
        else:
            valid_save_objs: list[str] = []
            train_save_objs: list[str] = other_objs
            for category, objs in category2obj.items():
                random_objs = random.sample(objs, len(objs))
                valid_save_objs.extend(random_objs[: args.num_valid_samples_per_category])
                train_save_objs.extend(random_objs[args.num_valid_samples_per_category :])

            valid_dir: Path = Path(f"{args.output_dir}/valid")
            valid_dir.mkdir(exist_ok=True)
            with valid_dir.joinpath("valid.jsonl").open("w") as f:
                f.write("\n".join(valid_save_objs) + "\n")

            random.shuffle(train_save_objs)
            train_dir: Path = Path(f"{args.output_dir}/train")
            train_dir.mkdir(exist_ok=True)
            with train_dir.joinpath("train.jsonl").open("w") as f:
                f.write("\n".join(train_save_objs) + "\n")

    preprocessor.make_multi_char_vocab(
        train_path=Path(f"{args.output_dir}/train/train.jsonl"),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
