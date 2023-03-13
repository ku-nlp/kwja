import copy
import io
import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Union

import pandas as pd
from rhoknp import BasePhrase, Document
from rhoknp.cohesion import Argument, ArgumentType, EndophoraArgument, ExophoraArgument, ExophoraReferent, Predicate

from kwja.utils.cohesion_analysis import is_bridging_target, is_coreference_target, is_pas_target

logger = logging.getLogger(__name__)


class Scorer:
    """A class to evaluate system output.

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        predicted_documents (List[Document]): システム予測文書集合
        gold_documents (List[Document]): 正解文書集合
        exophora_referents (List[ExophoraReferent]): 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentType を参照)
        pas_cases (List[str]): 評価の対象とする格 (rhoknp.cohesion.rel.CASE_TYPES を参照)
        pas_target (str): 述語項構造解析において述語として扱う対象 ('pred': 用言, 'noun': 体言, 'all': 両方, '': 述語なし (default: pred))
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)

    Attributes:
        doc_ids: (List[str]): 評価の対象となる文書の文書ID集合
        doc_id2predicted_document (Dict[str, Document]): 文書IDからシステム予測文書を引くための辞書
        doc_id2gold_document (Dict[str, Document]): 文書IDから正解文書を引くための辞書
        exophora_referents (List[ExophoraReferent]): 評価の対象とする外界照応の照応先
        pas_cases (List[str]): 評価の対象となる格
        pas_target (str): 述語項構造解析において述語として扱う対象
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        comp_result (Dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
        sub_scorers (List[SubScorer]): 文書ごとの評価を行うオブジェクトのリスト
    """

    ARGUMENT_TYPE2ANALYSIS = OrderedDict(
        [
            (ArgumentType.CASE_EXPLICIT, "overt"),
            (ArgumentType.CASE_HIDDEN, "dep"),
            (ArgumentType.OMISSION, "zero_endophora"),
            (ArgumentType.EXOPHORA, "zero_exophora"),
        ]
    )

    def __init__(
        self,
        predicted_documents: List[Document],
        gold_documents: List[Document],
        exophora_referents: List[ExophoraReferent],
        pas_cases: List[str],
        pas_target: str = "pred",
        bridging: bool = False,
        coreference: bool = False,
    ) -> None:
        # long document may have been ignored
        assert set(d.doc_id for d in predicted_documents) <= set(d.doc_id for d in gold_documents)
        self.doc_ids: List[str] = [d.doc_id for d in gold_documents]
        self.doc_id2predicted_document: Dict[str, Document] = {d.doc_id: d for d in predicted_documents}
        self.doc_id2gold_document: Dict[str, Document] = {d.doc_id: d for d in gold_documents}

        self.exophora_referents: List[ExophoraReferent] = exophora_referents
        self.pas_cases: List[str] = pas_cases if pas_target != "" else []
        self.pas_target: str = pas_target
        self.bridging: bool = bridging
        self.coreference: bool = coreference

        self.comp_result: Dict[tuple, str] = {}
        self.sub_scorers: List[SubScorer] = []

    def run(self) -> "ScoreResult":
        """読み込んだ正解文書集合とシステム予測文書集合に対して評価を行う

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result.clear()
        self.sub_scorers.clear()
        results = []
        for doc_id in self.doc_ids:
            sub_scorer = SubScorer(
                self.doc_id2predicted_document[doc_id],
                self.doc_id2gold_document[doc_id],
                exophora_referents=self.exophora_referents,
                pas_cases=self.pas_cases,
                pas_target=self.pas_target,
                bridging=self.bridging,
                coreference=self.coreference,
            )
            results.append(sub_scorer.run())
            self.sub_scorers.append(sub_scorer)
            self.comp_result.update({(doc_id, *k): v for k, v in sub_scorer.comp_result.items()})
        return reduce(add, results)


class SubScorer:
    """Scorer for single document pair.

    Args:
        predicted_document (Document): システム予測文書
        gold_document (Document): 正解文書
        exophora_referents (List[ExophoraReferent]): 評価の対象とする外界照応の照応先
        pas_cases (List[str]): 評価の対象とする格
        pas_target (str): 述語項構造解析において述語として扱う対象
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)

    Attributes:
        doc_id (str): 対象の文書ID
        predicted_document (Document): システム予測文書
        gold_document (Document): 正解文書
        exophora_referents (List[ExophoraReferent]): 評価の対象とする外界照応の照応先
        pas_cases (List[str]): 評価の対象となる格
        pas (bool): 述語項構造の評価を行うかどうか
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        predicted_pas_predicates: (List[Predicate]): システム予測文書に含まれる述語
        predicted_bridging_anaphors: (List[Predicate]): システム予測文書に含まれる橋渡し照応詞
        predicted_mentions: (List[BasePhrase]): システム予測文書に含まれるメンション
        gold_pas_predicates: (List[Predicate]): 正解文書に含まれる述語
        gold_bridging_anaphors: (List[Predicate]): 正解文書に含まれる橋渡し照応詞
        gold_mentions: (List[BasePhrase]): 正解文書に含まれるメンション
        comp_result (Dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
    """

    def __init__(
        self,
        predicted_document: Document,
        gold_document: Document,
        exophora_referents: List[ExophoraReferent],
        pas_cases: List[str],
        pas_target: str,
        bridging: bool,
        coreference: bool,
    ):
        assert predicted_document.doc_id == gold_document.doc_id
        self.doc_id: str = gold_document.doc_id
        self.predicted_document: Document = predicted_document
        self.gold_document: Document = gold_document

        self.exophora_referents: List[ExophoraReferent] = exophora_referents
        self.pas_cases: List[str] = pas_cases
        verbal = pas_target in ("pred", "all")
        nominal = pas_target in ("noun", "all")
        self.pas: bool = pas_target != ""
        self.bridging: bool = bridging
        self.coreference: bool = coreference

        self.predicted_pas_predicates: List[Predicate] = []
        self.predicted_bridging_anaphors: List[Predicate] = []
        self.predicted_mentions: List[BasePhrase] = []
        for base_phrase in predicted_document.base_phrases:
            if is_pas_target(base_phrase, verbal=verbal, nominal=nominal):
                self.predicted_pas_predicates.append(base_phrase.pas.predicate)
            if (self.bridging is True) and is_bridging_target(base_phrase):
                self.predicted_bridging_anaphors.append(base_phrase.pas.predicate)
            if (self.coreference is True) and is_coreference_target(base_phrase):
                self.predicted_mentions.append(base_phrase)

        self.gold_pas_predicates: List[Predicate] = []
        self.gold_bridging_anaphors: List[Predicate] = []
        self.gold_mentions: List[BasePhrase] = []
        for base_phrase in gold_document.base_phrases:
            if is_pas_target(base_phrase, verbal=verbal, nominal=nominal):
                self.gold_pas_predicates.append(base_phrase.pas.predicate)
            if (self.bridging is True) and is_bridging_target(base_phrase):
                self.gold_bridging_anaphors.append(base_phrase.pas.predicate)
            if (self.coreference is True) and is_coreference_target(base_phrase):
                self.gold_mentions.append(base_phrase)

        self.comp_result: Dict[tuple, str] = {}

    def run(self) -> "ScoreResult":
        """Perform evaluation for the given gold document and system prediction document.

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result.clear()
        pas_metrics = self._evaluate_pas() if self.pas is True else None
        bridging_metrics = self._evaluate_bridging() if self.bridging is True else None
        coreference_metric = self._evaluate_coreference() if self.coreference is True else None
        return ScoreResult(pas_metrics, bridging_metrics, coreference_metric)

    def _evaluate_pas(self) -> pd.DataFrame:
        """compute predicate-argument structure analysis scores"""
        metrics = pd.DataFrame(
            [[Metric() for _ in Scorer.ARGUMENT_TYPE2ANALYSIS.values()] for _ in self.pas_cases],
            index=self.pas_cases,
            columns=list(Scorer.ARGUMENT_TYPE2ANALYSIS.values()),
        )
        global_index2predicted_pas_predicate: Dict[int, Predicate] = {
            p.base_phrase.global_index: p for p in self.predicted_pas_predicates
        }
        global_index2gold_pas_predicate: Dict[int, Predicate] = {
            p.base_phrase.global_index: p for p in self.gold_pas_predicates
        }

        for global_index in range(len(self.predicted_document.base_phrases)):
            for pas_case in self.pas_cases:
                if global_index in global_index2predicted_pas_predicate:
                    predicted_pas_predicate = global_index2predicted_pas_predicate[global_index]
                    predicted_pas_arguments = predicted_pas_predicate.pas.get_arguments(pas_case, relax=False)
                    predicted_pas_arguments = self._filter_arguments(predicted_pas_arguments, predicted_pas_predicate)
                else:
                    predicted_pas_arguments = []
                # this project predicts one argument for one predicate
                assert len(predicted_pas_arguments) in (0, 1)

                if global_index in global_index2gold_pas_predicate:
                    gold_pas_predicate = global_index2gold_pas_predicate[global_index]
                    gold_pas_arguments = gold_pas_predicate.pas.get_arguments(pas_case, relax=False)
                    gold_pas_arguments = self._filter_arguments(gold_pas_arguments, gold_pas_predicate)
                    relaxed_gold_pas_arguments = gold_pas_predicate.pas.get_arguments(
                        pas_case, relax=True, include_nonidentical=True
                    )
                    if pas_case == "ガ":
                        relaxed_gold_pas_arguments += gold_pas_predicate.pas.get_arguments(
                            "判ガ", relax=True, include_nonidentical=True
                        )
                    relaxed_gold_pas_arguments = self._filter_arguments(relaxed_gold_pas_arguments, gold_pas_predicate)
                else:
                    gold_pas_arguments = relaxed_gold_pas_arguments = []

                key = (global_index, pas_case)

                # compute precision
                if len(predicted_pas_arguments) > 0:
                    predicted_pas_argument = predicted_pas_arguments[0]
                    if predicted_pas_argument in relaxed_gold_pas_arguments:
                        relaxed_gold_pas_argument = relaxed_gold_pas_arguments[
                            relaxed_gold_pas_arguments.index(predicted_pas_argument)
                        ]
                        # use argument_type of gold argument if possible
                        analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[relaxed_gold_pas_argument.type]
                        self.comp_result[key] = analysis
                        metrics.at[pas_case, analysis].tp += 1
                    else:
                        # system出力のargument_typeはgoldのものと違うので不整合が起きるかもしれない
                        analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[predicted_pas_argument.type]
                        self.comp_result[key] = "wrong"  # precision が下がる
                    metrics.at[pas_case, analysis].tp_fp += 1

                # compute recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
                if (len(gold_pas_arguments) > 0) or (
                    self.comp_result.get(key, None) in Scorer.ARGUMENT_TYPE2ANALYSIS.values()
                ):
                    recalled_pas_argument: Optional[Argument] = None
                    for relaxed_gold_pas_argument in relaxed_gold_pas_arguments:
                        if relaxed_gold_pas_argument in predicted_pas_arguments:
                            recalled_pas_argument = relaxed_gold_pas_argument  # 予測されている項を優先して正解の項に採用
                            break
                    if recalled_pas_argument is not None:
                        analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[recalled_pas_argument.type]
                        assert self.comp_result[key] == analysis
                    else:
                        # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用
                        analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[gold_pas_arguments[0].type]
                        if len(predicted_pas_arguments) > 0:
                            assert self.comp_result[key] == "wrong"
                        else:
                            self.comp_result[key] = "wrong"  # recall が下がる
                    metrics.at[pas_case, analysis].tp_fn += 1
        return metrics

    def _filter_arguments(self, arguments: List[Argument], predicate: Predicate) -> List[Argument]:
        filtered = []
        for argument in arguments:
            argument = copy.copy(argument)
            if argument.case.endswith("≒"):
                argument.case = argument.case[:-1]
            if argument.case == "判ガ":
                argument.case = "ガ"
            if argument.case == "ノ？":
                argument.case = "ノ"
            if isinstance(argument, ExophoraArgument):
                argument.exophora_referent.index = None  # 「不特定:人１」なども「不特定:人」として扱う
                if argument.exophora_referent not in self.exophora_referents:
                    continue
            else:
                assert isinstance(argument, EndophoraArgument)
                # filter out self-anaphora and cataphora
                if argument.base_phrase == predicate.base_phrase:
                    continue
                if (
                    # cataphora
                    argument.base_phrase.global_index > predicate.base_phrase.global_index
                    and argument.base_phrase.sentence.sid != predicate.base_phrase.sentence.sid
                ):
                    continue
            filtered.append(argument)
        return filtered

    def _evaluate_bridging(self) -> pd.Series:
        """compute bridging reference resolution scores"""
        metrics: Dict[str, Metric] = OrderedDict(
            (analysis, Metric()) for analysis in Scorer.ARGUMENT_TYPE2ANALYSIS.values()
        )
        global_index2predicted_anaphor: Dict[int, Predicate] = {
            p.base_phrase.global_index: p for p in self.predicted_bridging_anaphors
        }
        global_index2gold_anaphor: Dict[int, Predicate] = {
            p.base_phrase.global_index: p for p in self.gold_bridging_anaphors
        }

        for global_index in range(len(self.predicted_document.base_phrases)):
            if global_index in global_index2predicted_anaphor:
                predicted_anaphor = global_index2predicted_anaphor[global_index]
                predicted_antecedents: List[Argument] = self._filter_arguments(
                    predicted_anaphor.pas.get_arguments("ノ", relax=False), predicted_anaphor
                )
            else:
                predicted_antecedents = []
            # this project predicts one argument for one predicate
            assert len(predicted_antecedents) in (0, 1)

            if global_index in global_index2gold_anaphor:
                gold_anaphor: Predicate = global_index2gold_anaphor[global_index]
                gold_antecedents: List[Argument] = self._filter_arguments(
                    gold_anaphor.pas.get_arguments("ノ", relax=False), gold_anaphor
                )
                relaxed_gold_antecedents: List[Argument] = gold_anaphor.pas.get_arguments(
                    "ノ", relax=True, include_nonidentical=True
                )
                relaxed_gold_antecedents += gold_anaphor.pas.get_arguments("ノ？", relax=True, include_nonidentical=True)
                relaxed_gold_antecedents = self._filter_arguments(relaxed_gold_antecedents, gold_anaphor)
            else:
                gold_antecedents = relaxed_gold_antecedents = []

            key = (global_index, "ノ")

            # compute precision
            if len(predicted_antecedents) > 0:
                predicted_antecedent = predicted_antecedents[0]
                if predicted_antecedent in relaxed_gold_antecedents:
                    # use argument_type of gold antecedent if possible
                    relaxed_gold_antecedent = relaxed_gold_antecedents[
                        relaxed_gold_antecedents.index(predicted_antecedent)
                    ]
                    analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[relaxed_gold_antecedent.type]
                    if analysis == "overt":
                        analysis = "dep"
                    self.comp_result[key] = analysis
                    metrics[analysis].tp += 1
                else:
                    analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[predicted_antecedent.type]
                    if analysis == "overt":
                        analysis = "dep"
                    self.comp_result[key] = "wrong"
                metrics[analysis].tp_fp += 1

            # calculate recall
            if gold_antecedents or (self.comp_result.get(key, None) in Scorer.ARGUMENT_TYPE2ANALYSIS.values()):
                recalled_antecedent: Optional[Argument] = None
                for relaxed_gold_antecedent in relaxed_gold_antecedents:
                    if relaxed_gold_antecedent in predicted_antecedents:
                        recalled_antecedent = relaxed_gold_antecedent  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if recalled_antecedent is not None:
                    analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[recalled_antecedent.type]
                    if analysis == "overt":
                        analysis = "dep"
                    assert self.comp_result[key] == analysis
                else:
                    analysis = Scorer.ARGUMENT_TYPE2ANALYSIS[gold_antecedents[0].type]
                    if analysis == "overt":
                        analysis = "dep"
                    if len(predicted_antecedents) > 0:
                        assert self.comp_result[key] == "wrong"
                    else:
                        self.comp_result[key] = "wrong"
                metrics[analysis].tp_fn += 1
        return pd.Series(metrics)

    def _evaluate_coreference(self) -> pd.Series:
        """compute coreference resolution scores"""
        metric = Metric()
        global_index2predicted_mention: Dict[int, BasePhrase] = {p.global_index: p for p in self.predicted_mentions}
        global_index2gold_mention: Dict[int, BasePhrase] = {p.global_index: p for p in self.gold_mentions}
        for global_index in range(len(self.predicted_document.base_phrases)):
            if predicted_mention := global_index2predicted_mention.get(global_index):
                predicted_other_mentions = self._filter_mentions(predicted_mention.get_coreferents(), predicted_mention)
                predicted_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in predicted_mention.entities if e.exophora_referent is not None]
                )
            else:
                predicted_other_mentions = set()
                predicted_exophora_referents = set()

            if gold_mention := global_index2gold_mention.get(global_index):
                gold_other_mentions = self._filter_mentions(
                    gold_mention.get_coreferents(include_nonidentical=False),
                    gold_mention,
                )
                relaxed_gold_other_mentions = self._filter_mentions(
                    gold_mention.get_coreferents(include_nonidentical=True),
                    gold_mention,
                )
                gold_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in gold_mention.entities if e.exophora_referent is not None]
                )
                relaxed_gold_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in gold_mention.entities_all if e.exophora_referent is not None]
                )
            else:
                gold_other_mentions = relaxed_gold_other_mentions = set()
                gold_exophora_referents = relaxed_gold_exophora_referents = set()

            key = (global_index, "=")

            # compute precision
            if (len(predicted_other_mentions) > 0) or (len(predicted_exophora_referents) > 0):
                if (predicted_other_mentions & relaxed_gold_other_mentions) or (
                    predicted_exophora_referents & relaxed_gold_exophora_referents
                ):
                    self.comp_result[key] = "tp"
                    metric.tp += 1
                else:
                    self.comp_result[key] = "wrong"
                metric.tp_fp += 1

            # compute recall
            if (
                (len(gold_other_mentions) > 0)
                or (len(gold_exophora_referents) > 0)
                or (self.comp_result.get(key, None) == "tp")
            ):
                if (predicted_other_mentions & relaxed_gold_other_mentions) or (
                    predicted_exophora_referents & relaxed_gold_exophora_referents
                ):
                    assert self.comp_result[key] == "tp"
                else:
                    self.comp_result[key] = "wrong"
                metric.tp_fn += 1
        return pd.Series([metric], index=["all"])

    @staticmethod
    def _filter_mentions(other_mentions: Set[BasePhrase], mention: BasePhrase) -> Set[BasePhrase]:
        """filter out cataphora mentions"""
        return {
            another_mention for another_mention in other_mentions if another_mention.global_index < mention.global_index
        }

    def _filter_exophora_referents(self, exophora_referents: List[ExophoraReferent]) -> Set[str]:
        filtered = set()
        for exophora_referent in exophora_referents:
            exophora_referent = copy.copy(exophora_referent)
            exophora_referent.index = None
            if exophora_referent in self.exophora_referents:
                filtered.add(exophora_referent.text)
        return filtered


@dataclass(frozen=True)
class ScoreResult:
    """A data class for storing the numerical result of an evaluation"""

    pas_metrics: Optional[pd.DataFrame]
    bridging_metrics: Optional[pd.Series]
    coreference_metric: Optional[pd.Series]

    def to_dict(self) -> Dict[str, Dict[str, "Metric"]]:
        """convert data to dictionary"""
        df_all = pd.DataFrame(index=["all_case"])
        if self.pas is True:
            assert self.pas_metrics is not None
            df_pas: pd.DataFrame = self.pas_metrics.copy()
            df_pas["zero"] = df_pas["zero_endophora"] + df_pas["zero_exophora"]
            df_pas["dep_zero"] = df_pas["zero"] + df_pas["dep"]
            df_pas["pas"] = df_pas["dep_zero"] + df_pas["overt"]
            df_all = pd.concat([df_pas, df_all])
            df_all.loc["all_case"] = df_pas.sum(axis=0)

        if self.bridging is True:
            assert self.bridging_metrics is not None
            df_bar = self.bridging_metrics.copy()
            df_bar["zero"] = df_bar["zero_endophora"] + df_bar["zero_exophora"]
            df_bar["dep_zero"] = df_bar["zero"] + df_bar["dep"]
            assert df_bar["overt"] == Metric()  # No overt in BAR
            df_bar["pas"] = df_bar["dep_zero"]
            df_all.at["all_case", "bridging"] = df_bar["pas"]

        if self.coreference is True:
            assert self.coreference_metric is not None
            df_all.at["all_case", "coreference"] = self.coreference_metric["all"]

        return {
            rel: {analysis: metric for analysis, metric in analysis2metric.items() if pd.notnull(metric)}
            for rel, analysis2metric in df_all.to_dict(orient="index").items()
        }

    def export_txt(self, destination: Union[str, Path, TextIO]) -> None:
        """Export the evaluation results in a text format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
        """
        lines = []
        for rel, analysis2metric in self.to_dict().items():
            lines.append(f"{rel}格" if (self.pas_metrics is not None) and (rel in self.pas_metrics.index) else rel)
            for analysis, metric in analysis2metric.items():
                lines.append(f"  {analysis}")
                lines.append(f"    precision: {metric.precision:.4f} ({metric.tp}/{metric.tp_fp})")
                lines.append(f"    recall   : {metric.recall:.4f} ({metric.tp}/{metric.tp_fn})")
                lines.append(f"    F        : {metric.f1:.4f}")
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
        score_result_dict = self.to_dict()
        text += "case" + sep
        text += sep.join(score_result_dict["all_case"].keys()) + "\n"
        for rel, analysis2metric in score_result_dict.items():
            text += rel + sep
            text += sep.join(f"{metric.f1:.6}" for metric in analysis2metric.values())
            text += "\n"

        if isinstance(destination, str) or isinstance(destination, Path):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    @property
    def pas(self):
        """Whether self includes the score of predicate-argument structure analysis."""
        return self.pas_metrics is not None

    @property
    def bridging(self):
        """Whether self includes the score of bridging anaphora resolution."""
        return self.bridging_metrics is not None

    @property
    def coreference(self):
        """Whether self includes the score of coreference resolution."""
        return self.coreference_metric is not None

    def __add__(self, other: "ScoreResult") -> "ScoreResult":
        if self.pas is True:
            assert (self.pas_metrics is not None) and (other.pas_metrics is not None)
            pas_metrics = self.pas_metrics + other.pas_metrics
        else:
            pas_metrics = None
        if self.bridging is True:
            assert (self.bridging_metrics is not None) and (other.bridging_metrics is not None)
            bridging_metrics = self.bridging_metrics + other.bridging_metrics
        else:
            bridging_metrics = None
        if self.coreference is True:
            assert (self.coreference_metric is not None) and (other.coreference_metric is not None)
            coreference_metric = self.coreference_metric + other.coreference_metric
        else:
            coreference_metric = None
        return ScoreResult(pas_metrics, bridging_metrics, coreference_metric)


@dataclass
class Metric:
    """A data class to calculate and represent F-score"""

    tp_fp: int = 0
    tp_fn: int = 0
    tp: int = 0

    def __add__(self, other: "Metric"):
        return Metric(self.tp_fp + other.tp_fp, self.tp_fn + other.tp_fn, self.tp + other.tp)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self.tp_fp == other.tp_fp) and (self.tp_fn == other.tp_fn) and (self.tp == other.tp)

    @property
    def precision(self) -> float:
        if self.tp_fp == 0:
            return 0.0
        return self.tp / self.tp_fp

    @property
    def recall(self) -> float:
        if self.tp_fn == 0:
            return 0.0
        return self.tp / self.tp_fn

    @property
    def f1(self) -> float:
        if (self.tp_fp + self.tp_fn) == 0:
            return 0.0
        return (2 * self.tp) / (self.tp_fp + self.tp_fn)
