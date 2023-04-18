import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from rhoknp import KNP, Document, Sentence
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    args = parser.parse_args()

    knp = KNP()

    sid2knp_str: Dict[str, str] = {}
    num_errors: int = 0
    sid: int = 0
    with Path(args.input_path).open() as f:
        document: Document = Document.from_jumanpp(f.read())
        for sentence in tqdm(document.sentences):
            for morpheme in sentence.morphemes:
                if "代表表記" not in morpheme.semantics:
                    morpheme.semantics["代表表記"] = f"{morpheme.lemma}/{morpheme.reading}"
            try:
                knp_applied_sentence: Sentence = knp.apply_to_sentence(sentence)
            except ValueError:
                num_errors += 1
                continue
            sid2knp_str[f"{sid}"] = knp_applied_sentence.to_knp()
            sid += 1

    sid2knp_str_list: List[tuple[str, str]] = list(sid2knp_str.items())
    random.shuffle(sid2knp_str_list)
    train_size: int = int(len(sid2knp_str_list) * 0.9)
    valid_size: int = int(len(sid2knp_str_list) * 0.05)
    train_list: List[tuple[str, str]] = sid2knp_str_list[:train_size]
    valid_list: List[tuple[str, str]] = sid2knp_str_list[train_size : train_size + valid_size]
    test_list: List[tuple[str, str]] = sid2knp_str_list[train_size + valid_size :]

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"num_errors: {num_errors}")


if __name__ == "__main__":
    main()
