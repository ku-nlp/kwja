import logging
from collections import defaultdict
from dataclasses import dataclass

from rhoknp import BasePhrase, Document
from rhoknp.cohesion import Argument, ArgumentType, EndophoraArgument, ExophoraArgument, ExophoraReferent

from kwja.datamodule.extractors.base import Extractor, Phrase
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PasAnnotation:
    arguments_set: list[dict[str, list[str]]]


class PasExtractor(Extractor):
    def __init__(
        self,
        cases: list[str],
        exophors: list[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        self.pas_targets = ["pred", "noun"]
        super().__init__(exophors, restrict_target=restrict_target)
        self.cases = cases

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> PasAnnotation:
        bp_list = document.base_phrases
        arguments_set: list[dict[str, list[str]]] = [defaultdict(list) for _ in bp_list]
        for anaphor in [bp for sent in extract_target_sentences(document) for bp in sent.base_phrases]:
            if self.is_target(anaphor) is False:
                continue
            phrases[anaphor.global_index].is_target = True
            candidates: list[int] = [bp.global_index for bp in bp_list if self.is_candidate(bp, anaphor) is True]
            phrases[anaphor.global_index].candidates = candidates
            assert anaphor.pas is not None, "pas has not been set"
            for case in self.cases:
                arguments = anaphor.pas.get_arguments(case, relax=False)
                arguments_set[anaphor.global_index][case] = self._get_args(arguments, candidates)

        return PasAnnotation(arguments_set)

    def _get_args(
        self,
        orig_args: list[Argument],
        candidates: list[int],
    ) -> list[str]:
        """Get string representations of orig_args.
        If the return value is an empty list, do not calculate loss for this argument.
        overt: {dmid}%C
        case: {dmid}%N
        zero: {dmid}%O
        exophor: {exophor}
        no arg: [NULL]
        """
        # filter out non-target exophors
        args: list[Argument] = []
        for arg in orig_args:
            if isinstance(arg, ExophoraArgument):
                arg.exophora_referent = self._relax_exophora_referent(arg.exophora_referent)
                if arg.exophora_referent in self.exophora_referents:
                    args.append(arg)
                elif arg.exophora_referent.text == "[不明]":
                    return []  # don't train uncertain argument
            else:
                args.append(arg)
        if not args:
            return ["[NULL]"]
        arg_strings: list[str] = []
        for arg in args:
            if isinstance(arg, EndophoraArgument):
                if arg.base_phrase.global_index not in candidates:
                    logger.debug(f"argument: {arg} is not in candidates and ignored")
                    continue
                string = str(arg.base_phrase.global_index)
                if arg.type == ArgumentType.CASE_EXPLICIT:
                    string += "%C"
                elif arg.type == ArgumentType.CASE_HIDDEN:
                    string += "%N"
                else:
                    assert arg.type == ArgumentType.OMISSION
                    string += "%O"
            else:
                # exophora
                string = str(arg)
            arg_strings.append(string)
        return arg_strings

    def is_target(self, bp: BasePhrase) -> bool:
        return self.restrict_target is False or self.is_pas_target(
            bp,
            verbal=("pred" in self.pas_targets),
            nominal=("noun" in self.pas_targets),
        )

    @staticmethod
    def is_pas_target(bp: BasePhrase, verbal: bool, nominal: bool) -> bool:
        if verbal and "用言" in bp.features:
            return True
        if nominal and "非用言格解析" in bp.features:
            return True
        return False
