import logging
from collections import defaultdict
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
class PasAnnotation:
    arguments_set: list[dict[str, list[str]]]


class PasExtractor(Extractor):
    def __init__(
        self,
        cases: list[str],
        pas_targets: list[str],
        exophors: list[str],
        kc: bool = False,
    ) -> None:
        self.pas_targets = pas_targets
        super().__init__(exophors, kc)
        self.cases = cases

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> PasAnnotation:
        bp_list = document.bp_list()
        arguments_set: list[dict[str, list[str]]] = [defaultdict(list) for _ in bp_list]
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
                arguments = document.get_arguments(anaphor, relax=False)
                for case in self.cases:
                    arguments_set[anaphor.dtid][case] = self._get_args(
                        arguments[case], candidates
                    )

        return PasAnnotation(arguments_set)

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
                if arg.dep_type == "overt":
                    string += "%C"
                elif arg.dep_type == "dep":
                    string += "%N"
                else:
                    assert arg.dep_type in ("intra", "inter")
                    string += "%O"
            # exophor
            else:
                string = str(arg)
            arg_strings.append(string)
        return arg_strings

    def is_target(self, bp: BasePhrase) -> bool:
        return self.is_pas_target(
            bp,
            verbal=("pred" in self.pas_targets),
            nominal=("noun" in self.pas_targets),
        )

    @staticmethod
    def is_pas_target(bp: BasePhrase, verbal: bool, nominal: bool) -> bool:
        if verbal and "用言" in bp.tag.features:
            return True
        if nominal and "非用言格解析" in bp.tag.features:
            return True
        return False
