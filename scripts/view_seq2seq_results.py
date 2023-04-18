import difflib
import logging
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, cast

import jaconv
from rhoknp import Document, Jumanpp, Sentence
from tqdm.rich import tqdm

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
    def __init__(self, sys_sentences: List[Sentence], gold_sentences: List[Sentence]) -> None:
        self.tp: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)
        self.fp: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)
        self.fn: Dict[str, int] = dict(surf=0, reading=0, lemma=0, canon=0)

        self.num_same_texts: int = 0
        self.diffs: List[Diff] = self._search_diffs(sys_sentences, gold_sentences)

    @staticmethod
    def _convert(sentence: Sentence) -> List[str]:
        converteds: List[str] = []
        for mrph in sentence.morphemes:
            surf: str = jaconv.h2z(mrph.surf.replace("<unk>", "$"), ascii=True, digit=True)
            reading: str = jaconv.h2z(mrph.reading.replace("<unk>", "$"), ascii=True, digit=True)
            lemma: str = jaconv.h2z(mrph.lemma.replace("<unk>", "$"), ascii=True, digit=True)
            if mrph.canon is None or mrph.canon == "None":
                canon: str = f"{lemma}／{reading}"
            else:
                canon = jaconv.h2z(mrph.canon.replace("<unk>", "$"), ascii=True, digit=True)
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
        if pred_text == gold_text:
            self.num_same_texts += 1
        # else:
        #     print(f"{pred_text = }")
        #     print(f"{gold_text = }")
        return diff

    def _search_diffs(self, sys_sentences: List[Sentence], gold_sentences: List[Sentence]) -> List[Diff]:
        diffs = []
        for sys_sentence, gold_sentence in zip(sys_sentences, gold_sentences):
            diff: Diff = self._search_diff(self._convert(sys_sentence), self._convert(gold_sentence))
            diffs.append(diff)
        return diffs

    def compute_score(self, is_simple_output: bool = True) -> None:
        for diff in self.diffs:
            for p in diff:
                if p["diff_type"].equal:
                    true_keys: List[str] = ["surf", "reading", "lemma", "canon"]
                    false_keys: List[str] = []
                elif p["diff_type"].surf:
                    true_keys = []
                    false_keys = ["surf", "reading", "lemma", "canon"]
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
                    assert len(true_keys) + len(false_keys) == 4
                for key in true_keys:
                    self.tp[key] += len(p["sys_parts"])
                for key in false_keys:
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
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--typo-batch-size", type=int, default=1)
    parser.add_argument("--char-batch-size", type=int, default=1)
    parser.add_argument("--word-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    jumanpp = Jumanpp()
    # kwja = KWJA(
    #     options=[
    #         f"--model-size={args.model_size}",
    #         f"--device={args.device}",
    #         f"--typo-batch-size={args.typo_batch_size}",
    #         f"--char-batch-size={args.char_batch_size}",
    #         f"--word-batch-size={args.word_batch_size}",
    #         "--tasks=senter,char,word",
    #     ]
    # )

    sid_to_seq2seq_sent: Dict[str, Sentence] = dict()
    with Path(args.seq2seq_file).open() as f:
        seq2seq_document: Document = Document.from_jumanpp(f.read())
        for sentence in seq2seq_document.sentences:
            sid_to_seq2seq_sent[sentence.sid] = sentence

    output_dir: Path = Path(args.output_dir)
    for corpus in ["kyoto", "kwdlc", "fuman"]:
        sid_to_gold_sent: Dict[str, Sentence] = dict()
        sid_to_juman_sent: Dict[str, Sentence] = dict()
        sid_to_kwja_sent: Dict[str, Sentence] = dict()
        juman_dir: Path = output_dir / "juman" / corpus
        juman_dir.mkdir(parents=True, exist_ok=True)
        kwja_dir: Path = output_dir / "kwja" / corpus
        kwja_dir.mkdir(parents=True, exist_ok=True)

        gold_paths: List[Path] = list(Path(f"{args.dataset_dir}/{corpus}/test").glob("*.knp"))
        for gold_path in tqdm(gold_paths):
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

                # kwja_path: Path = kwja_dir / f"{gold_path.stem}_{gold_sentence.sid}.jumanpp"
                # if not kwja_path.exists():
                #     kwja_sentence: Sentence = kwja.apply_to_sentence(gold_sentence)
                #     with kwja_path.open("w") as f:
                #         f.write(kwja_sentence.to_jumanpp())
                # with kwja_path.open() as f:
                #     kwja_sentence = Sentence.from_knp(f.read())
                #     sid_to_kwja_sent[gold_sentence.sid] = kwja_sentence

        print(
            f"{corpus} (# of sents in juman/kwja/gold = {len(sid_to_juman_sent)}/{len(sid_to_kwja_sent)}/{len(sid_to_gold_sent)})"
        )

        jumans: List[Sentence] = []
        # kwjas: List[Sentence] = []
        seq2seqs: List[Sentence] = []
        golds: List[Sentence] = []
        for sid, gold_sent in sid_to_gold_sent.items():
            if sid not in sid_to_juman_sent or sid not in sid_to_seq2seq_sent:
                continue
            jumans.append(sid_to_juman_sent[sid])
            # kwjas.append(sid_to_kwja_sent[sid])
            seq2seqs.append(sid_to_seq2seq_sent[sid])
            golds.append(gold_sent)
            assert len(jumans) == len(seq2seqs) == len(golds)

        print("  jumanpp")
        juman_scorer = MorphologicalAnalysisScorer(jumans, golds)
        juman_scorer.compute_score()

        # print("  kwja")
        # kwja_scorer = MorphologicalAnalysisScorer(kwjas, golds)
        # kwja_scorer.compute_score()

        print("  seq2seq")
        system_scorer = MorphologicalAnalysisScorer(seq2seqs, golds)
        system_scorer.compute_score()

        print(f"# of same texts for seq2seq: {system_scorer.num_same_texts}")
        print(f"Ratio of same texts for seq2seq = {system_scorer.num_same_texts / len(seq2seqs) * 100:.2f}\n")


if __name__ == "__main__":
    main()
