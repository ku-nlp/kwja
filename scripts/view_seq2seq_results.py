import copy
import difflib
import logging
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jaconv
from rhoknp import Document, Sentence

logging.getLogger("rhoknp").setLevel(logging.ERROR)

PARTS_REGEX = re.compile(r"^([^_]+)_([^:]+):(.*)$")
OUTPUT_DIR: str = "./outputs"


@dataclass(frozen=True)
class DiffType:
    # equal, surf, {reading, lemma, pos, subpos}は排他的．
    equal: bool = False
    surf: bool = False
    reading: bool = False
    lemma: bool = False
    pos: bool = False
    subpos: bool = False
    canon: bool = False


class DiffPart(object):
    def __init__(self, diff_text: str):
        self.diff_text = diff_text
        self.has_diff: bool = False
        self.diff_part = self._separate_parts(diff_text)

    def __repr__(self):
        return f"{self.surf}_{self.reading}_{self.lemma}_{self.pos}_{self.subpos}_{self.canon}"

    def _separate_parts(self, text: str) -> tuple[str, str, str, str, str, str]:
        if text.startswith("! ") or text.startswith("+ ") or text.startswith("- "):
            self.has_diff = True
        return cast(tuple[str, str, str, str, str, str], tuple(text[2:].split("_")))

    @property
    def surf(self) -> str:
        return self.diff_part[0]

    @property
    def reading(self) -> str:
        return self.diff_part[1]

    @property
    def lemma(self) -> str:
        return self.diff_part[2]

    @property
    def pos(self) -> str:
        return self.diff_part[3]

    @property
    def subpos(self) -> str:
        return self.diff_part[4]

    @property
    def canon(self) -> str:
        return self.diff_part[5]


class Diff(object):
    __slots__ = ["data"]

    def __init__(self):
        self.data: list[dict] = []

    def __index__(self, index: int):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def append(self, diff_type: DiffType, sys_parts, gold_parts) -> None:
        if len(self.data) > 0 and self.data[-1]["diff_type"] == diff_type:
            self.data[-1]["sys_parts"].extend(sys_parts)
            self.data[-1]["gold_parts"].extend(gold_parts)
        else:
            self.data.append(dict(diff_type=diff_type, sys_parts=sys_parts, gold_parts=gold_parts))


class MorphologicalAnalysisScorer:
    def __init__(self, sys_sentences: list[Sentence], gold_sentences: list[Sentence]) -> None:
        self.tp: dict[str, int] = dict(surf=0, reading=0, lemma=0, pos=0, subpos=0, canon=0)
        self.fp: dict[str, int] = dict(surf=0, reading=0, lemma=0, pos=0, subpos=0, canon=0)
        self.fn: dict[str, int] = dict(surf=0, reading=0, lemma=0, pos=0, subpos=0, canon=0)

        self.num_same_texts: int = 0
        self.diffs: list[Diff] = self._search_diffs(sys_sentences, gold_sentences)

    @staticmethod
    def _convert(sentence: Sentence) -> list[str]:
        converteds: list[str] = []
        for mrph in sentence.morphemes:
            surf: str = jaconv.h2z(mrph.surf.replace("<unk>", "$"), ascii=True, digit=True)
            reading: str = jaconv.h2z(mrph.reading.replace("<unk>", "$"), ascii=True, digit=True)
            lemma: str = jaconv.h2z(mrph.lemma.replace("<unk>", "$"), ascii=True, digit=True)
            pos: str = jaconv.h2z(mrph.pos.replace("<unk>", "$"), ascii=True, digit=True)
            subpos: str = jaconv.h2z(mrph.subpos.replace("<unk>", "$"), ascii=True, digit=True)
            if mrph.canon is None or mrph.canon == "None":
                canon: str = f"{lemma}／{reading}"
            else:
                canon = jaconv.h2z(mrph.canon.replace("<unk>", "$"), ascii=True, digit=True)
            converteds.append(f"{surf}_{reading}_{lemma}_{pos}_{subpos}_{canon}")
        return converteds

    def _search_diff(self, pred: list[str], gold: list[str]) -> Diff:
        diff: Diff = Diff()
        max_len: int = max(len(pred), len(gold))
        lines = [x for x in difflib.context_diff(pred, gold, n=max_len, lineterm="")]
        if not lines:
            for p in pred:
                diff.append(DiffType(equal=True), [p], [p])
            return diff
        pred_with_diff: list[DiffPart] = []
        gold_with_diff: list[DiffPart] = []
        is_first_half: bool = True
        for idx, line in enumerate(lines):
            if idx < 4:
                continue
            if line.startswith("---"):
                is_first_half = False
            elif is_first_half:
                pred_with_diff.append(DiffPart(line))
            else:
                gold_with_diff.append(DiffPart(line))

        sys_idx: int = 0
        gold_idx: int = 0
        sys_len = len(pred)
        gold_len: int = len(gold)

        while sys_idx < sys_len and gold_idx < gold_len:
            if len(pred_with_diff) == 0:
                if gold_with_diff[gold_idx].has_diff:
                    diff.append(DiffType(surf=True), [], [gold[gold_idx]])
                    gold_idx += 1
                else:
                    diff.append(DiffType(equal=True), [pred[sys_idx]], [gold[gold_idx]])
                    sys_idx += 1
                    gold_idx += 1
            elif len(gold_with_diff) == 0:
                if pred_with_diff[sys_idx].has_diff:
                    diff.append(DiffType(surf=True), [pred[sys_idx]], [])
                    sys_idx += 1
                else:
                    diff.append(DiffType(equal=True), [pred[sys_idx]], [gold[gold_idx]])
                    sys_idx += 1
                    gold_idx += 1
            elif not pred_with_diff[sys_idx].has_diff and not gold_with_diff[gold_idx].has_diff:
                diff.append(DiffType(equal=True), [pred_with_diff[sys_idx]], [gold_with_diff[gold_idx]])
                sys_idx += 1
                gold_idx += 1
                continue
            else:
                sys_start_idx: int = sys_idx
                while sys_idx < len(pred_with_diff) and pred_with_diff[sys_idx].has_diff:
                    sys_idx += 1
                gold_start_idx: int = gold_idx
                while gold_idx < len(gold_with_diff) and gold_with_diff[gold_idx].has_diff:
                    gold_idx += 1

                pred_diff_parts: list[DiffPart] = pred_with_diff[sys_start_idx:sys_idx]
                gold_diff_parts: list[DiffPart] = gold_with_diff[gold_start_idx:gold_idx]
                pred_diff_idx, gold_diff_idx = 0, 0
                while pred_diff_idx < len(pred_diff_parts) and gold_diff_idx < len(gold_diff_parts):
                    pred_diff_part = pred_diff_parts[pred_diff_idx]
                    gold_diff_part = gold_diff_parts[gold_diff_idx]
                    if pred_diff_part.surf == gold_diff_part.surf:
                        diff_types = {
                            "equal": False,
                            "surf": False,
                            "reading": False,
                            "lemma": False,
                            "pos": False,
                            "subpos": False,
                            "canon": False,
                        }
                        if pred_diff_part.reading != gold_diff_part.reading:
                            diff_types["reading"] = True
                        if pred_diff_part.lemma != gold_diff_part.lemma:
                            diff_types["lemma"] = True
                        if pred_diff_part.canon != gold_diff_part.canon:
                            diff_types["canon"] = True

                        if pred_diff_part.pos == "未定義語" and gold_diff_part.pos == "名詞":
                            diff_types["subpos"] = True
                        elif pred_diff_part.pos != gold_diff_part.pos:
                            diff_types["pos"] = True
                        elif pred_diff_part.subpos != gold_diff_part.subpos:
                            diff_types["subpos"] = True
                        pred_diff_idx += 1
                        gold_diff_idx += 1
                        diff.append(DiffType(**diff_types), [pred_diff_part], [gold_diff_part])
                    else:
                        pred_surf_len = len(pred_diff_part.surf)
                        gold_surf_len = len(gold_diff_part.surf)
                        pred_diff_idx_start = pred_diff_idx
                        gold_diff_idx_start = gold_diff_idx
                        pred_diff_idx += 1
                        gold_diff_idx += 1
                        while pred_surf_len != gold_surf_len:
                            if pred_diff_idx_start + 1 < len(pred_diff_parts) and gold_diff_idx_start + 1 < len(
                                gold_diff_parts
                            ):
                                rest_pred_surf: str = "".join(x.surf for x in pred_diff_parts[pred_diff_idx:])
                                rest_gold_surf: str = "".join(x.surf for x in gold_diff_parts[gold_diff_idx:])
                                if rest_pred_surf == rest_gold_surf:
                                    break
                            if pred_surf_len > gold_surf_len:
                                if gold_diff_idx < len(gold_diff_parts):
                                    gold_surf_len += len(gold_diff_parts[gold_diff_idx].surf)
                                    gold_diff_idx += 1
                                else:
                                    break
                            else:
                                if pred_diff_idx < len(pred_diff_parts):
                                    pred_surf_len += len(pred_diff_parts[pred_diff_idx].surf)
                                    pred_diff_idx += 1
                                else:
                                    break
                        diff.append(
                            DiffType(surf=True),
                            pred_diff_parts[pred_diff_idx_start:pred_diff_idx],
                            gold_diff_parts[gold_diff_idx_start:gold_diff_idx],
                        )

        pred_text: str = "".join(x.split("_")[0] for x in pred)
        gold_text: str = "".join(x.split("_")[0] for x in gold)
        if pred_text == gold_text:
            self.num_same_texts += 1
        # else:
        #     print(f"{pred_text = }")
        #     print(f"{gold_text = }")
        return diff

    def _search_diffs(self, sys_sentences: list[Sentence], gold_sentences: list[Sentence]) -> list[Diff]:
        diffs = []
        for sys_sentence, gold_sentence in zip(sys_sentences, gold_sentences):
            diff: Diff = self._search_diff(self._convert(sys_sentence), self._convert(gold_sentence))
            diffs.append(diff)
        return diffs

    def compute_score(self, is_simple_output: bool = True) -> None:
        for diff in self.diffs:
            for p in diff:
                if p["diff_type"].equal:
                    true_keys: list[str] = ["surf", "pos", "subpos", "reading", "lemma", "canon"]
                    false_keys: list[str] = []
                elif p["diff_type"].surf:
                    true_keys = []
                    false_keys = ["surf", "pos", "subpos", "reading", "lemma", "canon"]
                else:
                    true_keys = ["surf"]
                    false_keys = []
                    if p["diff_type"].reading:
                        false_keys.append("reading")
                    else:
                        true_keys.append("reading")
                    if p["diff_type"].lemma:
                        false_keys.append("lemma")
                    else:
                        true_keys.append("lemma")
                    if p["diff_type"].canon:
                        false_keys.append("canon")
                    else:
                        true_keys.append("canon")
                    if p["diff_type"].pos:
                        false_keys.append("pos")
                        false_keys.append("subpos")
                    elif p["diff_type"].subpos:
                        true_keys.append("pos")
                        false_keys.append("subpos")
                    else:
                        true_keys.append("pos")
                        true_keys.append("subpos")
                    assert len(true_keys) + len(false_keys) == 6
                for key in true_keys:
                    self.tp[key] += len(p["sys_parts"])
                for key in false_keys:
                    self.fp[key] += len(p["sys_parts"])
                    self.fn[key] += len(p["gold_parts"])

        keys: list[str] = ["surf", "reading", "lemma", "canon"]
        outputs: list[str] = [" / ".join(keys)]
        output: list[str] = []
        for key in keys:
            prec: float = self.tp[key] / max(self.tp[key] + self.fp[key], 1)
            rec: float = self.tp[key] / max(self.tp[key] + self.fn[key], 1)
            f1: float = 2 * prec * rec / max(prec + rec, 1e-6)
            if is_simple_output:
                output.append(f"{f1 * 100:.2f}")
            else:
                output.append(f"{f1 * 100:.2f} (p = {prec * 100:.1f}, r = {rec * 100:.1f})")
        outputs.append(" / ".join(output))
        print("    " + " = ".join(outputs))

    def get_reading_errors(self) -> dict[int, dict[str, str]]:
        error_id2example: dict[int, dict[str, str]] = {}
        for sent_id, diff in enumerate(self.diffs):
            gold_text: str = ""
            sys_list: list[str] = []
            gold_list: list[str] = []
            for p in diff:
                gold_diff_part_text: str = "".join(str(m).split("_")[0] for m in p["gold_parts"])
                if p["diff_type"].reading:
                    gold_text += f"<span style='color:red'>{gold_diff_part_text}</span>"
                    sys_list.append("".join(str(m).split("_")[1] for m in p["sys_parts"]))
                    gold_list.append("".join(str(m).split("_")[1] for m in p["gold_parts"]))
                else:
                    gold_text += gold_diff_part_text
            if "<span " in gold_text:
                error_id2example[sent_id] = {
                    "text": gold_text,
                    "sys": "、".join(sys_list),
                    "gold": "、".join(gold_list),
                }
        return error_id2example

    def get_surf_errors(self) -> dict[int, dict[str, str]]:
        error_id2example: dict[int, dict[str, str]] = {}
        for sent_id, diff in enumerate(self.diffs):
            gold_text: str = ""
            sys_list: list[str] = []
            gold_list: list[str] = []
            for p in diff:
                gold_diff_part_text: str = "".join(str(m).split("_")[0] for m in p["gold_parts"])
                if p["diff_type"].surf:
                    gold_text += f"<span style='color:red'>{gold_diff_part_text}</span>"
                    sys_list.append("_".join(str(m).split("_")[0] for m in p["sys_parts"]))
                    gold_list.append("_".join(str(m).split("_")[0] for m in p["gold_parts"]))
                else:
                    gold_text += gold_diff_part_text
            if "<span " in gold_text:
                error_id2example[sent_id] = {
                    "text": gold_text,
                    "sys": "、".join(sys_list),
                    "gold": "、".join(gold_list),
                }
        return error_id2example


def format_juman(input_text: str) -> tuple[str, bool]:
    lines: list[str] = input_text.split("\n")
    dummies: list[str] = ["@", "@", "@", "@", "0", "@", "0", "@", "0", "@", "0", "NIL"]
    output_text: str = ""
    contain_dummy: bool = False
    for line in lines:
        if not line:
            continue
        if line == "EOS" or line.startswith("*") or line.startswith("+"):
            output_text += line + "\n"
        else:
            preds: list[str] = line.split(" ")
            if len(preds) == 4:
                mrphs: list[str] = copy.deepcopy(dummies)
                for idx in range(3):
                    mrphs[idx] = preds[idx]
                mrphs[-1] = f'"代表表記:{preds[3]}"' if preds[3] is not None else "NIL"
                output_text += " ".join(mrphs) + "\n"
            elif line in ["!!!!/!", "????/?", ",,,,/,"]:
                mrphs = copy.deepcopy(dummies)
                for idx in range(3):
                    mrphs[idx] = line[idx]
                mrphs[-1] = f'"代表表記:{line[-1]}/{line[-1]}"'
                output_text += " ".join(mrphs) + "\n"
            elif line == "............/...":
                mrphs = copy.deepcopy(dummies)
                for idx in range(3):
                    mrphs[idx] = "…"
                mrphs[-1] = '"代表表記:…/…"'
                output_text += " ".join(mrphs) + "\n"
            else:
                contain_dummy = True
                output_text += " ".join(dummies) + "\n"
    if "*" not in output_text:
        output_text = "*\n+\n" + output_text
    if not output_text.endswith("EOS\n"):
        output_text += "EOS\n"
    return output_text, contain_dummy


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str, required=True)
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-ck", "--compare-kwja", action="store_true")
    args = parser.parse_args()

    pred_dicts: dict[str, list[dict]] = dict()
    corpusid2text = {"0": "kyoto", "1": "kwdlc", "2": "fuman"}
    input_text: str = ""
    generated_text: str = ""
    corpus: str = ""
    with Path(args.input_file).open() as f:
        for line in f:
            if line[:3] == "src":
                corpus: str = corpusid2text[line[3]]
                input_text: str = line.strip()[6:]
            elif line == "EOS\n":
                generated_text += line
                if corpus not in pred_dicts:
                    pred_dicts[corpus] = []
                pred_dicts[corpus].append({"text": input_text, "generated": generated_text})
                input_text = ""
                generated_text = ""
            else:
                generated_text += line

    kwja_results: dict[str, dict[str, Sentence]] = dict()
    if args.compare_kwja:
        for corpus in ["kyoto", "kwdlc", "fuman"]:
            kwja_results[corpus] = dict()
            with Path(f"{OUTPUT_DIR}/predict_kwja/{corpus}_test.knp").open() as f:
                lines = [line for line in f]
            buffer: str = ""
            for line in lines:
                buffer += line
                if line == "EOS\n":
                    kwja_sentence: Sentence = Sentence.from_knp(buffer)
                    if kwja_sentence.text not in kwja_results[corpus]:
                        kwja_results[corpus][kwja_sentence.text] = kwja_sentence
                    buffer = ""

    for corpus in ["kyoto", "kwdlc", "fuman"]:
        print(f"{corpus}: (# of sentences = {len(pred_dicts[corpus])})")
        src_text2gold_sent: dict[str, Sentence] = dict()
        for path in Path(f"{args.dataset_dir}/{corpus}/test").glob("*.knp"):
            with path.open() as f:
                document: Document = Document.from_knp(f.read())
            for gold_sent in document.sentences:
                src_text2gold_sent[gold_sent.text] = gold_sent
        print(f"    # of sentences in gold: {len(src_text2gold_sent)}")

        src_text2juman_sent: dict[str, Sentence] = dict()
        for path in Path(f"{OUTPUT_DIR}/predict_juman_knp/{corpus}").glob("*.knp"):
            with path.open() as f:
                document: Document = Document.from_knp(f.read())
            for juman_sent in document.sentences:
                src_text2juman_sent[juman_sent.text] = juman_sent
        print(f"    # of sentences in juman: {len(src_text2juman_sent)}")

        jumans: list[Sentence] = []
        generators: list[Sentence] = []
        kwjas: list[Sentence] = []
        golds: list[Sentence] = []
        num_dummies: int = 0
        processed_generateds: set[str] = set()
        if args.compare_kwja:
            for pred in pred_dicts[corpus]:
                if pred["text"] not in kwja_results[corpus]:
                    continue
                if pred["generated"] in processed_generateds:
                    continue
                formatted, contain_dummy = format_juman(pred["generated"])
                num_dummies += 1 if contain_dummy else 0
                kwjas.append(kwja_results[corpus][pred["text"]])
                generators.append(Sentence.from_knp(formatted))
                jumans.append(src_text2juman_sent[pred["text"]])
                golds.append(src_text2gold_sent[pred["text"]])
                processed_generateds.add(pred["generated"])
            print(f"    # of sentences in kwja: {len(kwjas)}")
            assert len(jumans) == len(generators) == len(kwjas) == len(golds)
        else:
            for pred in pred_dicts[corpus]:
                if pred["generated"] in processed_generateds:
                    continue
                formatted, contain_dummy = format_juman(pred["generated"])
                num_dummies += 1 if contain_dummy else 0
                generators.append(Sentence.from_knp(formatted))
                jumans.append(src_text2juman_sent[pred["text"]])
                golds.append(src_text2gold_sent[pred["text"]])
                processed_generateds.add(pred["generated"])
            assert len(jumans) == len(generators) == len(golds)

        print("  jumanpp")
        juman_scorer = MorphologicalAnalysisScorer(jumans, golds)
        juman_scorer.compute_score()

        if args.compare_kwja:
            print("  kwja")
            kwja_scorer = MorphologicalAnalysisScorer(kwjas, golds)
            kwja_scorer.compute_score()

        print("  seq2seq")
        system_scorer = MorphologicalAnalysisScorer(generators, golds)
        system_scorer.compute_score()

        print(f"# of same texts for seq2seq: {system_scorer.num_same_texts}")
        print(
            f"Ratio of same texts for seq2seq = {system_scorer.num_same_texts / len(generators) * 100:.2f}\n"
        )


if __name__ == "__main__":
    main()
