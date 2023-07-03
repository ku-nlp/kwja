import json
import logging
import random
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Set, Union

from jinf import Jinf
from rhoknp import Morpheme, Sentence
from rhoknp.utils.reader import chunk_by_sentence
from tqdm import tqdm

from kwja.utils.logging_util import filter_logs

filter_logs(environment="production")
logging.basicConfig(format="")

logger = logging.getLogger("kwja_cli")
logger.setLevel(logging.INFO)

jinf = Jinf()

STOP_SURFS: Set[str] = {'"', "\u3000"}


def is_hiragana(value):
    return re.match(r"^[\u3040-\u309F\u30FC]+$", value) is not None


def get_canons(morpheme: Morpheme) -> List[str]:
    canons: List[str] = []
    if morpheme.canon is not None:
        canons.append(morpheme.canon)
    for feature in morpheme.features:
        if feature[:4] == "ALT-":
            canons.append(feature.split('"')[1].split(" ")[0].replace("代表表記:", ""))
    return canons


def set_canon(sentence: Sentence, morpheme_index: int) -> None:
    for morpheme in sentence.morphemes:
        if morpheme.index == morpheme_index:
            continue
        if morpheme.conjtype == "*" or morpheme.pos == "特殊":
            canon: str = f"{morpheme.lemma}/{morpheme.reading}"
        elif is_hiragana(morpheme.lemma):
            canon = f"{morpheme.lemma}/{morpheme.lemma}"
        else:
            canon_right: str = jinf(morpheme.reading, morpheme.conjtype, morpheme.conjform, "基本形")
            canon = f"{morpheme.lemma}/{canon_right}"
        morpheme.semantics["代表表記"] = canon


def get_is_excluded(sentence: Sentence) -> bool:
    is_excluded: bool = False
    for morpheme in sentence.morphemes:
        for stop_surf in STOP_SURFS:
            if stop_surf in morpheme.surf:
                is_excluded = True
                break
        if (morpheme.pos != "特殊") and (not is_hiragana(morpheme.reading)):
            is_excluded = True
            break
    return is_excluded


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="*", required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-max", "--max-samples", type=int, default=3)
    args = parser.parse_args()

    random.seed(42)

    sentences: List[Sentence] = []
    canon2freq: Dict[str, int] = {}
    excluded_nums: Dict[str, int] = {}
    for input_dir in args.input_dirs:
        for input_path in tqdm(Path(input_dir).glob("**/*.txt")):
            with input_path.open() as f:
                for knp in chunk_by_sentence(f):
                    try:
                        sentence: Sentence = Sentence.from_knp(knp)
                    except ValueError:
                        excluded_nums["sent_from_knp"] = excluded_nums.get("sent_from_knp", 0) + 1
                        continue
                    for morpheme in sentence.morphemes:
                        canons: List[str] = get_canons(morpheme)
                        if len(canons) > 1:
                            for canon in canons:
                                canon2freq[canon] = canon2freq.get(canon, 0) + 1
                    sentences.append(sentence)
    print(f"num_sentences: {len(sentences)}")

    sid2knp_str: Dict[str, str] = {}
    sid2changes: Dict[str, Dict[str, Union[int, dict[str, str]]]] = {}
    sampled_canon2freq: Dict[str, int] = {}
    for sentence in sentences:
        if get_is_excluded(sentence):
            continue
        for morpheme in sentence.morphemes:
            canons: List[str] = get_canons(morpheme)
            if len(canons) > 1 or sampled_canon2freq.get(morpheme.canon, 0) >= args.max_samples:
                continue

            if canon2freq.get(morpheme.canon, 0) >= 2:
                if morpheme.conjtype == "*":
                    surf: str = morpheme.reading
                    lemma: str = morpheme.reading
                else:
                    try:
                        lemma = jinf(morpheme.reading, morpheme.conjtype, morpheme.conjform, "基本形")
                        surf = jinf(lemma, morpheme.conjtype, "基本形", morpheme.conjform)
                    except (ValueError, NotImplementedError):
                        excluded_nums["jinf"] = excluded_nums.get("jinf", 0) + 1
                        continue
                if surf == morpheme.text and lemma == morpheme.lemma:
                    continue
                changes: Dict[str, Union[int, dict[str, str]]] = {
                    "morpheme_index": morpheme.index,
                    "before": {
                        "surf": morpheme.text,
                        "lemma": morpheme.lemma,
                    },
                    "after": {
                        "surf": surf,
                        "lemma": lemma,
                    },
                }
                morpheme.text = surf
                morpheme._text_escaped = surf
                morpheme.lemma = lemma
                try:
                    set_canon(sentence, morpheme_index=morpheme.index)
                except (ValueError, NotImplementedError):
                    excluded_nums["set_canon"] = excluded_nums.get("set_canon", 0) + 1
                    continue
                try:
                    sid2knp_str[sentence.sid] = sentence.to_knp()
                    sampled_canon2freq[morpheme.canon] = sampled_canon2freq.get(morpheme.canon, 0) + 1
                    sid2changes[sentence.sid] = changes
                    break
                except AttributeError:
                    excluded_nums["to_knp"] = excluded_nums.get("to_knp", 0) + 1
                    continue
        if len(sid2knp_str) >= 5000:
            break

    sid2knp_str_list: List[tuple[str, str]] = list(sid2knp_str.items())
    random.shuffle(sid2knp_str_list)
    train_size: int = int(len(sid2knp_str_list) * 0.9)
    valid_size: int = int(len(sid2knp_str_list) * 0.05)
    train_list: List[tuple[str, str]] = sid2knp_str_list[:train_size]
    valid_list: List[tuple[str, str]] = sid2knp_str_list[train_size : train_size + valid_size]
    test_list: List[tuple[str, str]] = sid2knp_str_list[train_size + valid_size :]

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "changes.json", "w") as f:
        json.dump(sid2changes, f, indent=2, ensure_ascii=False)

    train_dir: Path = output_dir / "train"
    if train_dir.exists():
        shutil.rmtree(str(train_dir))
    train_dir.mkdir(exist_ok=True)
    for name, knp_str in train_list:
        with (train_dir / f"{name}.knp").open("w") as f:
            f.write(f"{knp_str}\n")

    valid_dir: Path = output_dir / "valid"
    if valid_dir.exists():
        shutil.rmtree(str(valid_dir))
    valid_dir.mkdir(exist_ok=True)
    for name, knp_str in valid_list:
        with (valid_dir / f"{name}.knp").open("w") as f:
            f.write(f"{knp_str}\n")

    test_dir: Path = output_dir / "test"
    if test_dir.exists():
        shutil.rmtree(str(test_dir))
    test_dir.mkdir(exist_ok=True)
    for name, knp_str in test_list:
        with (test_dir / f"{name}.knp").open("w") as f:
            f.write(f"{knp_str}\n")

    print(f"train: {len(train_list)}")
    print(f"valid: {len(valid_list)}")
    print(f"test: {len(test_list)}")
    print(f"total: {len(train_list) + len(valid_list) + len(test_list)}")
    print(f"excluded_nums: {excluded_nums}")


if __name__ == "__main__":
    main()
