import difflib
import json
import logging
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast

import jaconv
from rhoknp import Document, Jumanpp, Morpheme, Sentence

logging.getLogger("rhoknp").setLevel(logging.ERROR)

PARTS_REGEX = re.compile(r"^([^_]+)_([^:]+):(.*)$")
OUTPUT_DIR: str = "./outputs"


@dataclass(frozen=True)
class DiffType:
    # equal, surf, {reading, lemma, canon}は排他的．
    equal: bool = False
    surf: bool = False
    reading: bool = False
    lemma: bool = False
    canon: bool = False


class DiffPart(object):
    def __init__(self, diff_text: str):
        self.diff_text = diff_text
        self.has_diff: bool = False
        self.diff_part = self._separate_parts(diff_text)

    def __repr__(self):
        return f"{self.surf}_{self.reading}_{self.lemma}_{self.canon}"

    def _separate_parts(self, text: str) -> Tuple[str, str, str, str]:
        if text.startswith("! ") or text.startswith("+ ") or text.startswith("- "):
            self.has_diff = True
        return cast(Tuple[str, str, str, str], tuple(text[2:].split("_")))

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
    def canon(self) -> str:
        return self.diff_part[3]


class Diff(object):
    __slots__ = ["data"]

    def __init__(self):
        self.data: List[Dict] = []

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
    def __init__(
        self,
        sys_sentences: List[Sentence],
        gold_sentences: List[Sentence],
        dataset_dir: Path,
        eval_norm: bool = False,
        eval_canon: bool = False,
    ) -> None:
        self.eval_norm: bool = eval_norm
        self.eval_canon: bool = eval_canon
        self.tp: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)
        self.fp: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)
        self.fn: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)
        self.sys_sentences: List[Sentence] = sys_sentences
        self.gold_sentences: List[Sentence] = gold_sentences

        with (dataset_dir / Path("canon/info.json")).open() as f:
            self.canon_info: Dict[str, Dict[str, Dict[str, str]]] = json.load(f)
        with (dataset_dir / Path("norm/info.json")).open() as f:
            self.norm_info: Dict[str, Dict[str, Dict[str, str]]] = json.load(f)

        self.num_diff_texts: int = 0
        self.norm_types: Set[str] = set()
        self.diffs: List[Diff] = self._search_diffs(sys_sentences, gold_sentences)

    def _convert(self, sentence: Sentence, norm_morphemes: List[Morpheme], canon_morpheme: List[Morpheme]) -> List[str]:
        converteds: List[str] = []
        norm_surfs: Set[str] = set(mrph.surf for mrph in norm_morphemes)
        canon_surfs: Set[str] = set(mrph.surf for mrph in canon_morpheme)
        for mrph in sentence.morphemes:
            surf: str = jaconv.h2z(mrph.surf.replace("<unk>", "$"), ascii=True, digit=True)
            if self.eval_norm and surf not in norm_surfs:
                continue
            if self.eval_canon and surf not in canon_surfs:
                continue
            reading: str = mrph.reading.replace("<unk>", "$")
            if "/" in mrph.reading and len(mrph.reading) > 1:
                reading = reading.split("/")[0]
            reading = jaconv.h2z(reading, ascii=True, digit=True)
            lemma: str = jaconv.h2z(mrph.lemma.replace("<unk>", "$"), ascii=True, digit=True)
            if mrph.canon is None or mrph.canon == "None":
                canon: str = f"{lemma}／{reading}"
            else:
                canon = mrph.canon.replace("<unk>", "$")
                canon_list: List[str] = canon.split("/")
                if len(canon_list) > 2 and canon_list[0] and canon_list[1]:
                    canon = f"{canon_list[0]}/{canon_list[1]}"
                canon = jaconv.h2z(canon, ascii=True, digit=True)
            converteds.append(f"{surf}_{reading}_{lemma}_{canon}")
        return converteds

    def _search_diff(self, pred: List[str], gold: List[str]) -> Diff:
        diff: Diff = Diff()
        max_len: int = max(len(pred), len(gold))
        lines = [x for x in difflib.context_diff(pred, gold, n=max_len, lineterm="")]
        if not lines:
            for p in pred:
                diff.append(DiffType(equal=True), [p], [p])
            return diff
        pred_with_diff: List[DiffPart] = []
        gold_with_diff: List[DiffPart] = []
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
        sys_len: int = len(pred)
        gold_len: int = len(gold)

        if sys_len == 0:
            for g in gold:
                diff.append(DiffType(surf=True), [], [g])
            return diff
        elif gold_len == 0:
            for p in pred:
                diff.append(DiffType(surf=True), [p], [])
            return diff

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

                pred_diff_parts: List[DiffPart] = pred_with_diff[sys_start_idx:sys_idx]
                gold_diff_parts: List[DiffPart] = gold_with_diff[gold_start_idx:gold_idx]
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
                            "canon": False,
                        }
                        if pred_diff_part.reading != gold_diff_part.reading:
                            diff_types["reading"] = True
                        if pred_diff_part.lemma != gold_diff_part.lemma:
                            diff_types["lemma"] = True
                        if pred_diff_part.canon != gold_diff_part.canon:
                            diff_types["canon"] = True

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
        if pred_text != gold_text:
            self.num_diff_texts += 1
            # print(f"{pred_text = }")
            # print(f"{gold_text = }")
        return diff

    def _search_diffs(self, sys_sentences: List[Sentence], gold_sentences: List[Sentence]) -> List[Diff]:
        diffs = []
        for sys_sentence, gold_sentence in zip(sys_sentences, gold_sentences):
            norm_morphemes: List[Morpheme] = []
            if self.eval_norm:
                for idx in self.norm_info[gold_sentence.sid]:
                    norm_morphemes.append(gold_sentence.morphemes[int(idx)])
            canon_morphemes: List[Morpheme] = []
            if self.eval_canon:
                for idx in self.canon_info[gold_sentence.sid]:
                    canon_morphemes.append(gold_sentence.morphemes[int(idx)])
            sys_converted: List[str] = self._convert(sys_sentence, norm_morphemes, canon_morphemes)
            gold_converted: List[str] = self._convert(gold_sentence, norm_morphemes, canon_morphemes)
            diff: Diff = self._search_diff(sys_converted, gold_converted)
            diffs.append(diff)
        return diffs

    def compute_score(self, is_simple_output: bool = True) -> None:
        for diff in self.diffs:
            for p in diff:
                correct_keys: Set[str] = set()
                if p["diff_type"].equal:
                    correct_keys.update({"surf", "reading", "lemma", "canon"})
                elif not p["diff_type"].surf:
                    correct_keys.add("surf")
                    if not p["diff_type"].reading:
                        correct_keys.add("reading")
                    if not p["diff_type"].lemma:
                        correct_keys.add("lemma")
                    if not p["diff_type"].canon:
                        correct_keys.add("canon")

                for key in ["surf", "reading", "lemma", "canon"]:
                    if key in correct_keys:
                        self.tp[key] += len(p["sys_parts"])
                    else:
                        self.fp[key] += len(p["sys_parts"])
                        self.fn[key] += len(p["gold_parts"])

        keys: List[str] = ["surf", "reading", "lemma", "canon"]
        outputs: List[str] = [" / ".join(keys)]
        output: List[str] = []
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--seq2seq-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    jumanpp = Jumanpp()

    sid_to_seq2seq_sent: Dict[str, Sentence] = dict()
    with Path(args.seq2seq_file).open() as f:
        seq2seq_document: Document = Document.from_jumanpp(f.read())
        for sentence in seq2seq_document.sentences:
            sid_to_seq2seq_sent[sentence.sid] = sentence

    output_dir: Path = Path(args.output_dir)
    for corpus in ["kyoto", "kwdlc", "fuman", "wac", "norm", "canon"]:
        sid_to_gold_sent: Dict[str, Sentence] = dict()
        sid_to_juman_sent: Dict[str, Sentence] = dict()
        juman_dir: Path = output_dir / "juman" / corpus
        juman_dir.mkdir(parents=True, exist_ok=True)

        gold_paths: List[Path] = list(Path(f"{args.dataset_dir}/{corpus}/test").glob("*.knp"))
        for gold_path in gold_paths:
            with gold_path.open() as f:
                gold_document: Document = Document.from_knp(f.read())
            for gold_sentence in gold_document.sentences:
                sid_to_gold_sent[gold_sentence.sid] = gold_sentence

                juman_path: Path = juman_dir / f"{gold_path.stem}_{gold_sentence.sid}.jumanpp"
                if not juman_path.exists():
                    juman_sentence: Sentence = jumanpp.apply_to_sentence(gold_sentence)
                    with juman_path.open("w") as f:
                        f.write(juman_sentence.to_jumanpp())
                with juman_path.open() as f:
                    juman_sentence = Sentence.from_jumanpp(f.read())
                    sid_to_juman_sent[gold_sentence.sid] = juman_sentence

        print(f"{corpus} (# of sents in juman/gold = {len(sid_to_juman_sent)}/{len(sid_to_gold_sent)})")

        jumans: List[Sentence] = []
        seq2seqs: List[Sentence] = []
        golds: List[Sentence] = []
        for sid, gold_sent in sid_to_gold_sent.items():
            if sid not in sid_to_juman_sent or sid not in sid_to_seq2seq_sent:
                continue
            jumans.append(sid_to_juman_sent[sid])
            seq2seqs.append(sid_to_seq2seq_sent[sid])
            golds.append(gold_sent)
            assert len(jumans) == len(seq2seqs) == len(golds)

        if corpus == "norm":
            print("  seq2seq")
            system_scorer = MorphologicalAnalysisScorer(seq2seqs, golds, dataset_dir=args.dataset_dir)
            system_scorer.compute_score()
            print(f"  # of different texts for seq2seq: {system_scorer.num_diff_texts}")
            print(
                f"  Ratio of same texts for seq2seq = {(len(seq2seqs) - system_scorer.num_diff_texts) / len(seq2seqs) * 100:.2f}"
            )
            print("  seq2seq (only target morpheme)")
            system_scorer = MorphologicalAnalysisScorer(seq2seqs, golds, dataset_dir=args.dataset_dir, eval_norm=True)
            system_scorer.compute_score()
            print()
        elif corpus == "canon":
            print("  seq2seq (only target morpheme)")
            system_scorer = MorphologicalAnalysisScorer(seq2seqs, golds, dataset_dir=args.dataset_dir, eval_canon=True)
            system_scorer.compute_score()
            print()
        else:
            print("  jumanpp")
            juman_scorer = MorphologicalAnalysisScorer(jumans, golds, dataset_dir=args.dataset_dir)
            juman_scorer.compute_score()

            print("  seq2seq")
            system_scorer = MorphologicalAnalysisScorer(seq2seqs, golds, dataset_dir=args.dataset_dir)
            system_scorer.compute_score()
            print(f"  # of different texts for seq2seq: {system_scorer.num_diff_texts}")
            print(
                f"  Ratio of same texts for seq2seq = {(len(seq2seqs) - system_scorer.num_diff_texts) / len(seq2seqs) * 100:.2f}\n"
            )


if __name__ == "__main__":
    main()
