import json
import logging
import random
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path

from jinf import Jinf
from rhoknp import KNP, Document, Sentence
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def is_hiragana(value):
    return re.match(r"^[\u3040-\u309F\u30FC]+$", value) is not None


def main():
    parser = ArgumentParser()
    parser.add_argument("-io", "--input-orig-path", type=str, required=True)
    parser.add_argument("-ij", "--input-juman-path", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(42)

    knp = KNP()
    jinf = Jinf()

    partials: list[dict[str, str]] = []
    partial: dict[str, str] = {}
    with Path(args.input_orig_path).open() as f:
        for line in f:
            if line := line.rstrip("\n"):
                if line.startswith("\t"):
                    annos = line.split("\t")
                    partial["text"] = partial.get("text", "") + annos[1]
                    partial["surf"] = annos[1]
                    for _anno_idx, anno in enumerate(annos[2:]):
                        if anno.startswith("baseform:"):
                            partial["lemma"] = anno.removeprefix("baseform:")
                        elif anno.startswith("conjtype:"):
                            partial["conjtype"] = anno.removeprefix("conjtype:")
                        elif anno.startswith("conjform:"):
                            partial["conjform"] = anno.removeprefix("conjform:")
                        else:
                            raise ValueError(f"This annotation type is not supported: {anno}")
                else:
                    partial["text"] = partial.get("text", "") + line
            else:
                partials.append(partial)
                partial = {}
        partials.append(partial)
    logger.info(f"Loaded {len(partials)} partials")

    sid2knp_str: dict[str, str] = {}
    sid2info: dict[str, dict[int, dict[str, str]]] = {}
    excluded_nums: dict[str, int] = {}
    with Path(args.input_juman_path).open() as f:
        document: Document = Document.from_jumanpp(f.read())
        for sentence_idx, sentence in enumerate(tqdm(document.sentences)):
            if sentence.text != partials[sentence_idx]["text"]:
                excluded_nums["text_mismatch"] = excluded_nums.get("text_mismatch", 0) + 1
                logger.warning(
                    f"Text mismatch: {sentence.text} != {partials[sentence_idx]['text']} at sentence index {sentence_idx}"
                )
                continue
            info: dict[int, dict[str, str]] = {}
            for morpheme in sentence.morphemes:
                pseudo_canon_type: str = ""
                if "代表表記" not in morpheme.semantics:
                    if morpheme.conjtype == "*":
                        canon: str = f"{morpheme.lemma}/{morpheme.reading}"
                        pseudo_canon_type = "活用なし"
                    elif is_hiragana(morpheme.lemma):
                        canon = f"{morpheme.lemma}/{morpheme.lemma}"
                        pseudo_canon_type = "かな"
                    else:
                        canon_right: str = jinf(morpheme.reading, morpheme.conjtype, morpheme.conjform, "基本形")
                        canon = f"{morpheme.lemma}/{canon_right}"
                        pseudo_canon_type = "活用"
                    morpheme.semantics["代表表記"] = canon
                if morpheme.surf == partials[sentence_idx]["surf"]:
                    info[morpheme.index] = {
                        "partial_annotation_type": "norm",
                        "surf": partials[sentence_idx]["surf"],
                        "lemma": partials[sentence_idx].get("lemma", ""),
                        "conjtype": partials[sentence_idx].get("conjtype", ""),
                        "conjform": partials[sentence_idx].get("conjform", ""),
                        "pseudo_canon_type": pseudo_canon_type,
                    }

            if len(info) == 0:
                excluded_nums["no_info"] = excluded_nums.get("no_info", 0) + 1
                continue

            try:
                knp_applied_sentence: Sentence = knp.apply_to_sentence(sentence)
            except ValueError:
                excluded_nums["knp_application_error"] = excluded_nums.get("knp_application_error", 0) + 1
                continue
            try:
                sid2knp_str[knp_applied_sentence.sid] = knp_applied_sentence.to_knp()
                assert len(info) > 0
                sid2info[knp_applied_sentence.sid] = info
            except AttributeError:
                excluded_nums["no_attribution_error"] = excluded_nums.get("no_attribution_error", 0) + 1
                continue

    sid2knp_str_list: list[tuple[str, str]] = list(sid2knp_str.items())
    random.shuffle(sid2knp_str_list)
    train_size: int = int(len(sid2knp_str_list) * 0.9)
    valid_size: int = int(len(sid2knp_str_list) * 0.05)
    train_list: list[tuple[str, str]] = sid2knp_str_list[:train_size]
    valid_list: list[tuple[str, str]] = sid2knp_str_list[train_size : train_size + valid_size]
    test_list: list[tuple[str, str]] = sid2knp_str_list[train_size + valid_size :]

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "info.json", "w") as f:
        json.dump(sid2info, f, indent=2, ensure_ascii=False)

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
    print(f"num_errors: {excluded_nums}")


if __name__ == "__main__":
    main()
