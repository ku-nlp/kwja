import logging
from dataclasses import dataclass

from rhoknp import BasePhrase, Document
from rhoknp.rel import Argument, ExophoraReferent, SpecialArgument
from rhoknp.rel.pas import BaseArgument

from .base import Extractor, Phrase

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class BridgingAnnotation:
    arguments_set: list[list[str]]


class BridgingExtractor(Extractor):
    def __init__(
        self,
        bar_rels: list[str],
        exophors: list[ExophoraReferent],
        kc: bool = False,
    ) -> None:
        super().__init__(exophors, kc)
        self.rels = bar_rels

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> BridgingAnnotation:
        bp_list = document.base_phrases
        arguments_set: list[list[str]] = [[] for _ in bp_list]
        for sentence in document.sentences:
            for anaphor in sentence.base_phrases:
                is_target_phrase: bool = (
                    self.is_target(anaphor)
                    and self._kc_skip_sentence(sentence, document) is False
                )
                phrases[anaphor.global_index].is_target = is_target_phrase
                if is_target_phrase is False:
                    continue
                candidates: list[int] = [
                    bp.global_index
                    for bp in bp_list
                    if self.is_candidate(bp, anaphor) is True
                ]
                phrases[anaphor.global_index].candidates = candidates
                arguments: list[BaseArgument] = []
                for rel in self.rels:
                    arguments += anaphor.pas.get_arguments(rel, relax=False)
                arguments_set[anaphor.global_index] = self._get_args(
                    arguments, candidates
                )

        return BridgingAnnotation(arguments_set)

    def _get_args(
        self,
        orig_args: list[BaseArgument],
        candidates: list[int],
    ) -> list[str]:
        """Get string representations of orig_args.
        If the return value is an empty list, do not calculate loss for this argument.
        overt: {dmid}%C
        case: {dmid}%N
        zero: {dmid}%O
        exophor: {exophor}
        no arg: NULL
        """
        # filter out non-target exophors
        args: list[BaseArgument] = []
        for arg in orig_args:
            if isinstance(arg, SpecialArgument):
                arg.exophora_referent = self._relax_exophora_referent(
                    arg.exophora_referent
                )
                if arg.exophora_referent in self.exophora_referents:
                    args.append(arg)
                elif arg.exophora_referent == ExophoraReferent("[不明]"):
                    return []  # don't train uncertain argument
            else:
                args.append(arg)
        if not args:
            return ["[NULL]"]
        arg_strings: list[str] = []
        for arg in args:
            if isinstance(arg, Argument):
                if arg.base_phrase.global_index not in candidates:
                    logger.debug(f"argument: {arg} is not in candidates and ignored")
                    continue
                string = str(arg.base_phrase.global_index)
            else:
                # exophora
                string = str(arg)
            arg_strings.append(string)
        return arg_strings

    def is_target(self, bp: BasePhrase) -> bool:
        return self.is_bridging_target(bp)

    @staticmethod
    def is_bridging_target(bp: BasePhrase) -> bool:
        return bp.features.get("体言") is True and bp.features.get("非用言格解析") is not True