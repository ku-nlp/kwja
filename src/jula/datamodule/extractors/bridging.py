import logging
from dataclasses import dataclass

from kyoto_reader import (
    UNCERTAIN,
    Argument,
    BaseArgument,
    BasePhrase,
    Document,
    SpecialArgument,
)

from .base import Extractor, Phrase

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class BridgingAnnotation:
    arguments_set: list[list[str]]


class BridgingExtractor(Extractor):
    def __init__(
        self,
        bar_rels: list[str],
        exophors: list[str],
        kc: bool = False,
    ) -> None:
        super().__init__(exophors, kc)
        self.rels = bar_rels

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> BridgingAnnotation:
        bp_list = document.bp_list()
        arguments_set: list[list[str]] = [[] for _ in bp_list]
        for sentence in document:
            for anaphor in sentence.bps:
                is_target_phrase: bool = (
                    self.is_target(anaphor)
                    and self._kc_skip_sentence(sentence, document) is False
                )
                phrases[anaphor.dtid].is_target = is_target_phrase
                if is_target_phrase is False:
                    continue
                candidates: list[int] = [
                    bp.dtid for bp in bp_list if self.is_candidate(bp, anaphor) is True
                ]
                phrases[anaphor.dtid].candidates = candidates
                arguments: dict[str, list[BaseArgument]] = document.get_arguments(
                    anaphor, relax=False
                )
                args: list[BaseArgument] = sum(
                    (arguments[rel] for rel in self.rels), []
                )
                arguments_set[anaphor.dtid] = self._get_args(args, candidates)

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
                if exophor := self.relax_exophors.get(arg.exophor):
                    arg.exophor = exophor
                    args.append(arg)
                elif arg.exophor == UNCERTAIN:
                    return []  # don't train uncertain argument
            else:
                args.append(arg)
        if not args:
            return ["NULL"]
        arg_strings: list[str] = []
        for arg in args:
            if isinstance(arg, Argument):
                if arg.dtid not in candidates:
                    logger.debug(f"argument: {arg} is not in candidates and ignored")
                    continue
                string = str(arg.dtid)
            # exophor
            else:
                string = str(arg)
            arg_strings.append(string)
        return arg_strings

    def is_target(self, bp: BasePhrase) -> bool:
        return self.is_bridging_target(bp)

    @staticmethod
    def is_bridging_target(bp: BasePhrase) -> bool:
        return "体言" in bp.tag.features and "非用言格解析" not in bp.tag.features
