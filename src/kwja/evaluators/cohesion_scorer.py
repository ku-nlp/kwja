import argparse
import io
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import pandas as pd
from rhoknp import BasePhrase, Document
from rhoknp.cohesion import Argument, ArgumentType, EndophoraArgument, ExophoraArgument, ExophoraReferent, Predicate

from kwja.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    """A class to evaluate system output.

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`kyoto_reader.Document`

    Args:
        documents_pred (list[Document]): システム予測文書集合
        documents_gold (list[Document]): 正解文書集合
        target_cases (list[str]): 評価の対象とする格 (kyoto_reader.ALL_CASES を参照)
        exophora_referents (list[ExophoraReferent]): 評価の対象とする外界照応の照応先 (kyoto_reader.ALL_EXOPHORS を参照)
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)
        pas_target (str): 述語項構造解析において述語として扱う対象 ('pred': 用言, 'noun': 体言, 'all': 両方, '': 述語なし (default: pred))

    Attributes:
        cases (list[str]): 評価の対象となる格
        doc_ids: (list[str]): 評価の対象となる文書の文書ID集合
        did2document_pred (dict[str, Document]): 文書IDからシステム予測文書を引くための辞書
        did2document_gold (dict[str, Document]): 文書IDから正解文書を引くための辞書
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        pas_target (str): 述語項構造解析において述語として扱う対象
        comp_result (dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
        sub_scorers (list[SubScorer]): 文書ごとの評価を行うオブジェクトのリスト
        exophora_referents (list[ExophoraReferent]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
    """

    DEPTYPE2ANALYSIS = OrderedDict(
        [
            (ArgumentType.CASE_EXPLICIT, "overt"),
            (ArgumentType.CASE_HIDDEN, "dep"),
            (ArgumentType.OMISSION, "zero_endophora"),
            (ArgumentType.EXOPHORA, "zero_exophora"),
        ]
    )

    def __init__(
        self,
        documents_pred: list[Document],
        documents_gold: list[Document],
        target_cases: list[str],
        exophora_referents: list[ExophoraReferent],
        bridging: bool = False,
        coreference: bool = False,
        pas_target: str = "pred",
    ) -> None:
        # long document may have been ignored
        assert set(doc.doc_id for doc in documents_pred) <= set(doc.doc_id for doc in documents_gold)
        self.cases: list[str] = target_cases if pas_target != "" else []
        self.doc_ids: list[str] = [doc.doc_id for doc in documents_pred]
        self.did2document_pred: dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: dict[str, Document] = {doc.doc_id: doc for doc in documents_gold}
        self.bridging: bool = bridging
        self.coreference: bool = coreference
        self.pas_target: str = pas_target

        self.comp_result: dict[tuple, str] = {}
        self.sub_scorers: list[SubScorer] = []
        self.exophora_referents: list[ExophoraReferent] = exophora_referents

    def run(self) -> "ScoreResult":
        """読み込んだ正解文書集合とシステム予測文書集合に対して評価を行う

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result = {}
        self.sub_scorers = []
        all_results = []
        for doc_id in self.doc_ids:
            sub_scorer = SubScorer(
                self.did2document_pred[doc_id],
                self.did2document_gold[doc_id],
                cases=self.cases,
                bridging=self.bridging,
                coreference=self.coreference,
                exophora_referents=self.exophora_referents,
                pas_target=self.pas_target,
            )
            all_results.append(sub_scorer.run())
            self.sub_scorers.append(sub_scorer)
            self.comp_result.update({(doc_id, *key): val for key, val in sub_scorer.comp_result.items()})
        return reduce(add, all_results)


class SubScorer:
    """Scorer for single document pair.

    Args:
        document_pred (Document): システム予測文書
        document_gold (Document): 正解文書
        cases (list[str]): 評価の対象とする格
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)
        exophora_referents (list[ExophoraReferent]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
        pas_target (str): 述語項構造解析において述語として扱う対象

    Attributes:
        doc_id (str): 対象の文書ID
        document_pred (Document): システム予測文書
        document_gold (Document): 正解文書
        cases (list[str]): 評価の対象となる格
        pas (bool): 述語項構造の評価を行うかどうか
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        comp_result (dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
        exophora_referents (list[ExophoraReferent]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
        predicates_pred: (list[Predicate]): システム予測文書に含まれる述語
        bridgings_pred: (list[Predicate]): システム予測文書に含まれる橋渡し照応詞
        mentions_pred: (list[BasePhrase]): システム予測文書に含まれるメンション
        predicates_gold: (list[Predicate]): 正解文書に含まれる述語
        bridgings_gold: (list[Predicate]): 正解文書に含まれる橋渡し照応詞
        mentions_gold: (list[BasePhrase]): 正解文書に含まれるメンション
    """

    def __init__(
        self,
        document_pred: Document,
        document_gold: Document,
        cases: list[str],
        bridging: bool,
        coreference: bool,
        exophora_referents: list[ExophoraReferent],
        pas_target: str,
    ):
        assert document_pred.doc_id == document_gold.doc_id
        self.doc_id: str = document_gold.doc_id
        self.document_pred: Document = document_pred
        self.document_gold: Document = document_gold
        self.cases: list[str] = cases
        self.pas: bool = pas_target != ""
        self.bridging: bool = bridging
        self.coreference: bool = coreference
        self.comp_result: dict[tuple, str] = {}
        self.exophora_referents: list[ExophoraReferent] = exophora_referents

        self.predicates_pred: list[Predicate] = []
        self.bridgings_pred: list[Predicate] = []
        self.mentions_pred: list[BasePhrase] = []
        for bp in document_pred.base_phrases:
            assert bp.pas is not None, "pas has not been set"
            if PasExtractor.is_pas_target(
                bp,
                verbal=(pas_target in ("pred", "all")),
                nominal=(pas_target in ("noun", "all")),
            ):
                self.predicates_pred.append(bp.pas.predicate)
            if self.bridging and BridgingExtractor.is_bridging_target(bp):
                self.bridgings_pred.append(bp.pas.predicate)
            if self.coreference and CoreferenceExtractor.is_coreference_target(bp):
                self.mentions_pred.append(bp)
        self.predicates_gold: list[Predicate] = []
        self.bridgings_gold: list[Predicate] = []
        self.mentions_gold: list[BasePhrase] = []
        for bp in document_gold.base_phrases:
            assert bp.pas is not None, "pas has not been set"
            if PasExtractor.is_pas_target(
                bp,
                verbal=(pas_target in ("pred", "all")),
                nominal=(pas_target in ("noun", "all")),
            ):
                self.predicates_gold.append(bp.pas.predicate)
            if self.bridging and BridgingExtractor.is_bridging_target(bp):
                self.bridgings_gold.append(bp.pas.predicate)
            if self.coreference and CoreferenceExtractor.is_coreference_target(bp):
                self.mentions_gold.append(bp)

    def run(self) -> "ScoreResult":
        """Perform evaluation for the given gold document and system prediction document.

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result = {}
        measures_pas = self._evaluate_pas() if self.pas else None
        measures_bridging = self._evaluate_bridging() if self.bridging else None
        measure_coref = self._evaluate_coref() if self.coreference else None
        return ScoreResult(measures_pas, measures_bridging, measure_coref)

    def _evaluate_pas(self) -> pd.DataFrame:
        """calculate predicate-argument structure analysis scores"""
        measures = pd.DataFrame(
            [[Measure() for _ in Scorer.DEPTYPE2ANALYSIS.values()] for _ in self.cases],
            index=self.cases,
            columns=Scorer.DEPTYPE2ANALYSIS.values(),
        )
        global_index2predicate_pred: dict[int, Predicate] = {
            pred.base_phrase.global_index: pred for pred in self.predicates_pred
        }
        global_index2predicate_gold: dict[int, Predicate] = {
            pred.base_phrase.global_index: pred for pred in self.predicates_gold
        }

        for global_index in range(len(self.document_pred.base_phrases)):
            for case in self.cases:
                if global_index in global_index2predicate_pred:
                    predicate_pred = global_index2predicate_pred[global_index]
                    args_pred = predicate_pred.pas.get_arguments(case, relax=False)
                else:
                    args_pred = []
                # this project predicts one argument for one predicate
                assert len(args_pred) in (0, 1)

                if global_index in global_index2predicate_gold:
                    predicate_gold = global_index2predicate_gold[global_index]
                    args_gold = predicate_gold.pas.get_arguments(case, relax=False)
                    args_gold = self._filter_args(args_gold, predicate_gold)
                    args_gold_relaxed = predicate_gold.pas.get_arguments(case, relax=True)
                    if case == "ガ":
                        args_gold_relaxed += predicate_gold.pas.get_arguments("判ガ", relax=True)
                else:
                    args_gold = args_gold_relaxed = []

                key = (global_index, case)

                # calculate precision
                if args_pred:
                    arg = args_pred[0]
                    if arg in args_gold_relaxed:
                        # use dep_type of gold argument if possible
                        arg_gold_prec = args_gold_relaxed[args_gold_relaxed.index(arg)]
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg_gold_prec.type]
                        self.comp_result[key] = analysis
                        measures.at[case, analysis].correct += 1
                    else:
                        # system出力のdep_typeはgoldのものと違うので不整合が起きるかもしれない
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg.type]
                        self.comp_result[key] = "wrong"  # precision が下がる
                    measures.at[case, analysis].denom_pred += 1

                # calculate recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
                # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用
                if args_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                    arg_gold_rec: Optional[Argument] = None
                    for arg in args_gold_relaxed:
                        if arg in args_pred:
                            arg_gold_rec = arg  # 予測されている項を優先して正解の項に採用
                            break
                    if arg_gold_rec is not None:
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg_gold_rec.type]
                        assert self.comp_result[key] == analysis
                    else:
                        analysis = Scorer.DEPTYPE2ANALYSIS[args_gold[0].type]
                        if args_pred:
                            assert self.comp_result[key] == "wrong"
                        else:
                            self.comp_result[key] = "wrong"  # recall が下がる
                    measures.at[case, analysis].denom_gold += 1
        return measures

    def _filter_args(self, args: list[Argument], predicate: Predicate) -> list[Argument]:
        filtered_args = []
        for arg in args:
            if isinstance(arg, ExophoraArgument):
                if arg.exophora_referent not in self.exophora_referents:  # filter out non-target exophors
                    continue
                arg.exophora_referent.index = None  # 「不特定:人１」なども「不特定:人」として扱う
            else:
                assert isinstance(arg, EndophoraArgument)
                # filter out self-anaphora and cataphora
                if predicate.base_phrase == arg.base_phrase:
                    continue
                if (
                    predicate.base_phrase.global_index < arg.base_phrase.global_index
                    and arg.base_phrase.sentence.sid != predicate.base_phrase.sentence.sid
                ):
                    continue
            filtered_args.append(arg)
        return filtered_args

    def _evaluate_bridging(self) -> pd.Series:
        """calculate bridging anaphora resolution scores"""
        measures: dict[str, Measure] = OrderedDict((anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values())
        global_index2anaphor_pred: dict[int, Predicate] = {
            pred.base_phrase.global_index: pred for pred in self.bridgings_pred
        }
        global_index2anaphor_gold: dict[int, Predicate] = {
            pred.base_phrase.global_index: pred for pred in self.bridgings_gold
        }

        for global_index in range(len(self.document_pred.base_phrases)):
            if global_index in global_index2anaphor_pred:
                anaphor_pred = global_index2anaphor_pred[global_index]
                antecedents_pred: list[Argument] = self._filter_args(
                    anaphor_pred.pas.get_arguments("ノ", relax=False), anaphor_pred
                )
            else:
                antecedents_pred = []
            # this project predicts one argument for one predicate
            assert len(antecedents_pred) in (0, 1)

            if global_index in global_index2anaphor_gold:
                anaphor_gold: Predicate = global_index2anaphor_gold[global_index]
                antecedents_gold: list[Argument] = self._filter_args(
                    anaphor_gold.pas.get_arguments("ノ", relax=False), anaphor_gold
                )
                antecedents_gold_relaxed: list[Argument] = anaphor_gold.pas.get_arguments("ノ", relax=True)
                antecedents_gold_relaxed += anaphor_gold.pas.get_arguments("ノ？", relax=True)
                antecedents_gold_relaxed = self._filter_args(antecedents_gold_relaxed, anaphor_gold)
            else:
                antecedents_gold = antecedents_gold_relaxed = []

            key = (global_index, "ノ")

            # calculate precision
            if antecedents_pred:
                antecedent_pred = antecedents_pred[0]
                if antecedent_pred in antecedents_gold_relaxed:
                    # use dep_type of gold antecedent if possible
                    antecedent_gold_prec = antecedents_gold_relaxed[antecedents_gold_relaxed.index(antecedent_pred)]
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold_prec.type]
                    if analysis == "overt":
                        analysis = "dep"
                    self.comp_result[key] = analysis
                    measures[analysis].correct += 1
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_pred.type]
                    if analysis == "overt":
                        analysis = "dep"
                    self.comp_result[key] = "wrong"
                measures[analysis].denom_pred += 1

            # calculate recall
            if antecedents_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                antecedent_gold_rec: Optional[Argument] = None
                for ant in antecedents_gold_relaxed:
                    if ant in antecedents_pred:
                        antecedent_gold_rec = ant  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if antecedent_gold_rec is not None:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold_rec.type]
                    if analysis == "overt":
                        analysis = "dep"
                    assert self.comp_result[key] == analysis
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedents_gold[0].type]
                    if analysis == "overt":
                        analysis = "dep"
                    if antecedents_pred:
                        assert self.comp_result[key] == "wrong"
                    else:
                        self.comp_result[key] = "wrong"
                measures[analysis].denom_gold += 1
        return pd.Series(measures)

    def _evaluate_coref(self) -> pd.Series:
        """calculate coreference resolution scores"""
        measure = Measure()
        for global_index in range(len(self.document_pred.base_phrases)):
            src_mention_pred = self.document_pred.base_phrases[global_index]
            tgt_mentions_pred = self.filter_mentions(src_mention_pred.get_coreferents(), src_mention_pred)
            exophors_pred = {
                e.exophora_referent.text for e in src_mention_pred.entities if e.exophora_referent is not None
            }

            src_mention_gold = self.document_gold.base_phrases[global_index]
            tgt_mentions_gold = self.filter_mentions(
                src_mention_gold.get_coreferents(include_nonidentical=False),
                src_mention_gold,
            )
            tgt_mentions_gold_relaxed = self.filter_mentions(
                src_mention_gold.get_coreferents(include_nonidentical=True),
                src_mention_gold,
            )
            exophors_gold = {
                e.exophora_referent.text for e in src_mention_gold.entities if e.exophora_referent is not None
            }
            exophors_gold_relaxed = {
                e.exophora_referent.text for e in src_mention_gold.entities_all if e.exophora_referent is not None
            }

            key = (global_index, "=")

            # calculate precision
            if tgt_mentions_pred or exophors_pred:
                if (tgt_mentions_pred & tgt_mentions_gold_relaxed) or (exophors_pred & exophors_gold_relaxed):
                    self.comp_result[key] = "correct"
                    measure.correct += 1
                else:
                    self.comp_result[key] = "wrong"
                measure.denom_pred += 1

            # calculate recall
            if tgt_mentions_gold or exophors_gold or (self.comp_result.get(key, None) == "correct"):
                if (tgt_mentions_pred & tgt_mentions_gold_relaxed) or (exophors_pred & exophors_gold_relaxed):
                    assert self.comp_result[key] == "correct"
                else:
                    self.comp_result[key] = "wrong"
                measure.denom_gold += 1
        return pd.Series([measure], index=["all"])

    @staticmethod
    def filter_mentions(tgt_mentions: set[BasePhrase], src_mention: BasePhrase) -> set[BasePhrase]:
        """filter out cataphors"""
        return {tgt_mention for tgt_mention in tgt_mentions if tgt_mention.global_index < src_mention.global_index}


@dataclass(frozen=True)
class ScoreResult:
    """A data class for storing the numerical result of an evaluation"""

    measures_pas: Optional[pd.DataFrame]
    measures_bridging: Optional[pd.Series]
    measure_coref: Optional[pd.Series]

    def to_dict(self) -> dict[str, dict[str, "Measure"]]:
        """convert data to dictionary"""
        df_all = pd.DataFrame(index=["all_case"])
        if self.pas:
            assert self.measures_pas is not None
            df_pas: pd.DataFrame = self.measures_pas.copy()
            df_pas["zero"] = df_pas["zero_endophora"] + df_pas["zero_exophora"]
            df_pas["dep_zero"] = df_pas["zero"] + df_pas["dep"]
            df_pas["pas"] = df_pas["dep_zero"] + df_pas["overt"]
            df_all = pd.concat([df_pas, df_all])
            df_all.loc["all_case"] = df_pas.sum(axis=0)

        if self.bridging:
            assert self.measures_bridging is not None
            df_bar = self.measures_bridging.copy()
            df_bar["zero"] = df_bar["zero_endophora"] + df_bar["zero_exophora"]
            df_bar["dep_zero"] = df_bar["zero"] + df_bar["dep"]
            assert df_bar["overt"] == Measure()  # No overt in BAR
            df_bar["pas"] = df_bar["dep_zero"]
            df_all.at["all_case", "bridging"] = df_bar["pas"]

        if self.coreference:
            assert self.measure_coref is not None
            df_all.at["all_case", "coreference"] = self.measure_coref["all"]

        return {
            k1: {k2: v2 for k2, v2 in v1.items() if pd.notnull(v2)} for k1, v1 in df_all.to_dict(orient="index").items()
        }

    def export_txt(self, destination: Union[str, Path, TextIO]) -> None:
        """Export the evaluation results in a text format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
        """
        lines = []
        for key, ms in self.to_dict().items():
            lines.append(f"{key}格" if self.measures_pas is not None and key in self.measures_pas.index else key)
            for analysis, measure in ms.items():
                lines.append(f"  {analysis}")
                lines.append(f"    precision: {measure.precision:.4f} ({measure.correct}/{measure.denom_pred})")
                lines.append(f"    recall   : {measure.recall:.4f} ({measure.correct}/{measure.denom_gold})")
                lines.append(f"    F        : {measure.f1:.4f}")
        text = "\n".join(lines) + "\n"

        if isinstance(destination, str) or isinstance(destination, Path):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def export_csv(self, destination: Union[str, Path, TextIO], sep: str = ",") -> None:
        """Export the evaluation results in a csv format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
            sep (str): 区切り文字 (default: ',')
        """
        text = ""
        result_dict = self.to_dict()
        text += "case" + sep
        text += sep.join(result_dict["all_case"].keys()) + "\n"
        for case, measures in result_dict.items():
            text += case + sep
            text += sep.join(f"{measure.f1:.6}" for measure in measures.values())
            text += "\n"

        if isinstance(destination, str) or isinstance(destination, Path):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    @property
    def pas(self):
        """Whether self includes the score of predicate-argument structure analysis."""
        return self.measures_pas is not None

    @property
    def bridging(self):
        """Whether self includes the score of bridging anaphora resolution."""
        return self.measures_bridging is not None

    @property
    def coreference(self):
        """Whether self includes the score of coreference resolution."""
        return self.measure_coref is not None

    def __add__(self, other: "ScoreResult") -> "ScoreResult":
        if self.pas:
            assert self.measures_pas is not None and other.measures_pas is not None
            measures_pas = self.measures_pas + other.measures_pas
        else:
            measures_pas = None
        if self.bridging:
            assert self.measures_bridging is not None and other.measures_bridging is not None
            measures_bridging = self.measures_bridging + other.measures_bridging
        else:
            measures_bridging = None
        if self.coreference:
            assert self.measure_coref is not None and other.measure_coref is not None
            measure_coref = self.measure_coref + other.measure_coref
        else:
            measure_coref = None
        return ScoreResult(measures_pas, measures_bridging, measure_coref)


@dataclass
class Measure:
    """A data class to calculate and represent F-measure"""

    denom_pred: int = 0
    denom_gold: int = 0
    correct: int = 0

    def __add__(self, other: "Measure"):
        return Measure(
            self.denom_pred + other.denom_pred,
            self.denom_gold + other.denom_gold,
            self.correct + other.correct,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Measure):
            return False
        return (
            self.denom_pred == other.denom_pred
            and self.denom_gold == other.denom_gold
            and self.correct == other.correct
        )

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            return 0.0
        return self.correct / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            return 0.0
        return self.correct / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            return 0.0
        return 2 * self.correct / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default=None,
        type=str,
        help="path to directory where system output KWDLC files exist (default: None)",
    )
    parser.add_argument(
        "--gold-dir",
        default=None,
        type=str,
        help="path to directory where gold KWDLC files exist (default: None)",
    )
    parser.add_argument(
        "--coreference",
        "--coref",
        "--cr",
        action="store_true",
        default=False,
        help="perform coreference resolution",
    )
    parser.add_argument(
        "--bridging",
        "--brg",
        "--bar",
        action="store_true",
        default=False,
        help="perform bridging anaphora resolution",
    )
    parser.add_argument(
        "--case-string",
        type=str,
        default="ガ,ヲ,ニ,ガ２",
        help='case strings separated by ","',
    )
    parser.add_argument(
        "--exophors",
        "--exo",
        type=str,
        default="著者,読者,不特定:人,不特定:物",
        help='exophor strings separated by ","',
    )
    parser.add_argument(
        "--read-prediction-from-pas-tag",
        action="store_true",
        default=False,
        help="use <述語項構造:> tag instead of <rel > tag in prediction files",
    )
    parser.add_argument(
        "--pas-target",
        choices=["", "pred", "noun", "all"],
        default="pred",
        help="PAS analysis evaluation target (pred: verbal predicates, noun: nominal predicates)",
    )
    parser.add_argument(
        "--result-csv",
        default=None,
        type=str,
        help="path to csv file which prediction result is exported (default: None)",
    )
    args = parser.parse_args()

    documents_pred = [Document.from_knp(path.read_text()) for path in Path(args.prediction_dir).glob("*.knp")]
    documents_gold = [Document.from_knp(path.read_text()) for path in Path(args.gold_dir).glob("*.knp")]

    msg = (
        '"ノ" found in case string. If you want to perform bridging anaphora resolution, specify "--bridging" '
        "option instead"
    )
    assert "ノ" not in args.case_string.split(","), msg
    scorer = Scorer(
        documents_pred,
        documents_gold,
        target_cases=args.case_string.split(","),
        exophora_referents=[ExophoraReferent(e) for e in args.exophors.split(",")],
        coreference=args.coreference,
        bridging=args.bridging,
        pas_target=args.pas_target,
    )
    result = scorer.run()
    if args.result_csv:
        result.export_csv(args.result_csv)
    result.export_txt(sys.stdout)


if __name__ == "__main__":
    main()
