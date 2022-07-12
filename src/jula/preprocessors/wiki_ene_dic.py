import json
import pickle
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import dartsclone
from tqdm import tqdm

BAD_ENE_PATTERNS: list[str] = [
    r"^1\.7\.19\.",  # 芸術作品名
    r"^1\.7\.20\.0$",  # 出版物名＿その他
    r"^1\.7\.20\.2$",  # 雑誌名
    r"^1\.7\.23\.",  # 称号名
    r"^1\.7\.25\.",  # 単位名
    r"^1\.8\.",  # バーチャルアドレス名
    r"^2\.",  # 時間表現 (Timex)
    r"^3\.",  # 数値表現 (Numex)
    "^9$",  # IGNORED
]


class WikiDicPreprocessor:
    def __init__(self, input_json_path: str, output_dir: str):
        self.input_json_path: str = input_json_path
        self.output_dir: str = output_dir
        Path(self.output_dir).mkdir(exist_ok=True)

        self.title_pattern = re.compile(r"^(.+) \([^)]+\)$")
        self.bad_title_pattern = re.compile(
            r"^\d+$|^[\u0000-\u007F]{1,2}$|^[\u3040-\u30FF]{1,2}$"
        )  # numbers, two alphabets, two hiragana/katakanas
        self.bad_ene_pattern = re.compile("|".join(BAD_ENE_PATTERNS))
        self.ene_pattern = re.compile(r"^(0$|\d+\.\d+)")
        self.stop_titles: set[str] = {"ぼく", "ぼくら", "先生"}

        self.entries: list[dict] = self._load_json(path=input_json_path)

    @staticmethod
    def _load_json(path: str) -> list[dict]:
        entries: list[dict] = []
        with Path(path).open() as f:
            for line in tqdm(f, desc="load original json file..."):
                entries.append(json.loads(line))
        return entries

    def _is_bad_entry(self, entry: dict) -> bool:
        if len(entry["title_normalized"]) <= 1:
            return True
        if entry["title_normalized"] in self.stop_titles:
            return True
        if self.bad_title_pattern.match(entry["title_normalized"]) is not None:
            return True
        for cand in list(entry["ENEs"].values())[0]:
            # ENE = Extended Named Entity
            # Please refer to Project SHINRA (http://ene-project.info/ene8-1/)
            if self.bad_ene_pattern.match(cand["ENE_id"]) is not None:
                return True
        return False

    def _normalize_enes(self, entry: dict) -> list[str]:
        enes = set()
        for cand in list(entry["ENEs"].values())[0]:
            m = self.ene_pattern.match(cand["ENE_id"])
            if m is None:
                sys.stderr.write("malformed ENE: {}\n".format(cand["ENE_id"]))
                continue
            else:
                enes.add(m[1])
        return list(enes)

    def get_filtered_entries(self, save_filtered_results: bool = False) -> list[dict]:
        converted_entries: dict[str, set] = {}
        for entry in tqdm(self.entries, desc="process filtering..."):
            m = self.title_pattern.match(entry["title"])
            entry["title_normalized"] = entry["title"] if m is None else m[1]
            if self._is_bad_entry(entry):
                continue
            entry["ENEs_normalized"] = self._normalize_enes(entry)
            if entry["title_normalized"] in converted_entries:
                for ene in entry["ENEs_normalized"]:
                    converted_entries[entry["title_normalized"]].add(ene)
            else:
                converted_entries[entry["title_normalized"]] = set(entry["ENEs_normalized"])

        outputs: list[dict[str, Union[str, list[str]]]] = [
            {"title": k, "classes": list(v)} for k, v in converted_entries.items()
        ]
        if save_filtered_results:
            with Path(f"{self.output_dir}/filtered.jsonl").open(mode="w") as f:
                for output in outputs:
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
        return outputs

    def build_db(self, entries: list[dict]) -> None:
        ene_set_all: set = set()
        title2class: dict[bytes, int] = dict()
        classes_all: dict[str, int] = dict()
        for entry in entries:
            title_byte: bytes = entry["title"].encode()
            ene_set_all |= set(entry["classes"])
            class_str: str = "\t".join(sorted(entry["classes"]))
            if class_str not in classes_all:
                classes_all[class_str] = len(classes_all)
            title2class[title_byte] = classes_all[class_str]

        sorted_titles: list[bytes] = sorted(title2class.keys())
        sorted_classes: list[int] = [title2class[t] for t in sorted_titles]

        # build index of darts-clone
        darts = dartsclone.DoubleArray()
        darts.build(sorted_titles, values=sorted_classes)
        darts.save(f"{self.output_dir}/wiki.da")

        # save classes with darts-clone's index
        values: list[list[str]] = [[""]] * len(classes_all)
        for k, v in classes_all.items():
            values[v] = k.split("\t")
        with Path(f"{self.output_dir}/wiki_values.pkl").open("wb") as f:
            f.write(pickle.dumps(values))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-json-path",
        type=str,
        help="path to input json file obtained from Project SHINRA",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data",
        help="path to directory to save",
    )
    parser.add_argument(
        "-s",
        "--save-filtered-results",
        action="store_true",
        help="whether to create an intermediate file to save the filtering results",
    )
    args = parser.parse_args()

    wikidic_preprocessor = WikiDicPreprocessor(
        input_json_path=args.input_json_path,
        output_dir=args.output_dir,
    )
    entries = wikidic_preprocessor.get_filtered_entries(save_filtered_results=args.save_filtered_results)
    wikidic_preprocessor.build_db(entries=entries)


if __name__ == "__main__":
    main()
