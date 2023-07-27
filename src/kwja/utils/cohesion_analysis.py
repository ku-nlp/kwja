import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from rhoknp import BasePhrase, Morpheme
from rhoknp.cohesion import Argument, EndophoraArgument, ExophoraArgument, ExophoraReferent

from kwja.utils.sub_document import is_target_sentence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohesionBasePhrase:
    """A wrapper class of BasePhrase for cohesion analysis"""

    base_phrase: BasePhrase
    is_target: bool
    antecedent_candidates: Optional[List[BasePhrase]] = None
    # case -> argument_tags / "=" -> mention_tags
    rel2tags: Optional[Dict[str, List[str]]] = None

    def __getattr__(self, attr_name: str):
        if attr_name in {"is_target", "antecedent_candidates", "rel2tags"}:
            return getattr(self, attr_name)
        else:
            return getattr(self.base_phrase, attr_name)


class CohesionUtils:
    def __init__(self, restrict_target: bool) -> None:
        self.restrict_target = restrict_target

    @property
    def rels(self) -> List[str]:
        raise NotImplementedError

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        raise NotImplementedError

    @staticmethod
    def is_antecedent_candidate(antecedent: Union[BasePhrase, Morpheme], anaphor: Union[BasePhrase, Morpheme]) -> bool:
        raise NotImplementedError

    def get_antecedent_candidate_morphemes(self, morpheme: Morpheme, morphemes: List[Morpheme]) -> List[Morpheme]:
        return [m for m in morphemes if self.is_antecedent_candidate(m, morpheme)]


class PasUtils(CohesionUtils):
    def __init__(
        self,
        cases: List[str],
        target: str,
        exophora_referents: List[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        super().__init__(restrict_target)
        self.cases = cases
        self.target = target
        self.extractor = PasExtractor(
            cases,
            [er.type for er in exophora_referents],
            verbal_predicate=self.target in ("pred", "all"),
            nominal_predicate=self.target in ("noun", "all"),
        )

    @property
    def rels(self) -> List[str]:
        return self.cases

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases: List[CohesionBasePhrase] = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.extractor.is_target(base_phrase):
                antecedent_candidates = self.extractor.get_candidates(base_phrase)
                all_arguments = self.extractor.extract_rels(base_phrase)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={
                            case: self._get_argument_tags(all_arguments[case], antecedent_candidates)
                            for case in self.cases
                        },
                    )
                )
            else:
                cohesion_base_phrases.append(CohesionBasePhrase(base_phrase, is_target=False))
        return cohesion_base_phrases

    def _get_argument_tags(self, arguments: List[Argument], antecedent_candidates: List[BasePhrase]) -> List[str]:
        """Get argument tags.

        Note:
            If the return value is an empty list, do not compute loss for the argument.

            endophora argument: string of base phrase global index
            exophora argument: exophora referent
            no argument: [NULL]
        """
        target_arguments: List[Argument] = []
        for argument in arguments:
            argument = copy.copy(argument)  # not to overwrite the gold exophora_referent
            if isinstance(argument, EndophoraArgument):
                if argument.base_phrase in antecedent_candidates:
                    target_arguments.append(argument)
                else:
                    logger.info(f'argument "{argument}" is ignored ({argument.sentence.sent_id})')
            else:  # ExophoraArgument
                argument.exophora_referent.index = None  # 不特定:人１ -> 不特定:人
                target_arguments.append(argument)

        argument_tags: List[str] = []
        for target_argument in target_arguments:
            if isinstance(target_argument, EndophoraArgument):
                argument_tag = str(target_argument.base_phrase.global_index)
            else:
                assert isinstance(target_argument, ExophoraArgument)
                argument_tag = f"[{target_argument.exophora_referent.text}]"  # 著者 -> [著者]
            argument_tags.append(argument_tag)
        return argument_tags or ["[NULL]"]


class BridgingUtils(CohesionUtils):
    def __init__(
        self,
        cases: List[str],
        exophora_referents: List[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        super().__init__(restrict_target)
        self.cases: List[str] = cases
        self.extractor = BridgingExtractor(cases, [er.type for er in exophora_referents])

    @property
    def rels(self) -> List[str]:
        return self.cases

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases: List[CohesionBasePhrase] = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.extractor.is_target(base_phrase):
                all_arguments = self.extractor.extract_rels(base_phrase)
                antecedent_candidates = self.extractor.get_candidates(base_phrase)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={
                            case: self._get_argument_tags(all_arguments[case], antecedent_candidates)
                            for case in self.cases
                        },
                    )
                )
            else:
                cohesion_base_phrases.append(CohesionBasePhrase(base_phrase, is_target=False))
        return cohesion_base_phrases

    def _get_argument_tags(self, arguments: List[Argument], antecedent_candidates: List[BasePhrase]) -> List[str]:
        """Get argument tags.

        Note:
            If the return value is an empty list, do not compute loss for the argument.

            endophora argument: {base_phrase_global_index}
            exophora argument: {exophora argument}
            no argument: [NULL]
        """
        target_arguments: List[Argument] = []
        for argument in arguments:
            argument = copy.copy(argument)  # not to overwrite the gold exophora_referent
            if isinstance(argument, EndophoraArgument):
                if argument.base_phrase in antecedent_candidates:
                    target_arguments.append(argument)
                else:
                    logger.info(f'argument "{argument}" is ignored ({argument.sentence.sent_id})')
            else:  # ExophoraArgument
                argument.exophora_referent.index = None  # 不特定:人１ -> 不特定:人
                target_arguments.append(argument)

        argument_tags: List[str] = []
        for target_argument in target_arguments:
            if isinstance(target_argument, EndophoraArgument):
                argument_tag = str(target_argument.base_phrase.global_index)
            else:
                assert isinstance(target_argument, ExophoraArgument)
                argument_tag = f"[{target_argument.exophora_referent.text}]"  # 著者 -> [著者]
            argument_tags.append(argument_tag)
        return argument_tags or ["[NULL]"]


class CoreferenceUtils(CohesionUtils):
    def __init__(self, exophora_referents: List[ExophoraReferent], restrict_target: bool) -> None:
        super().__init__(restrict_target)
        self.extractor = CoreferenceExtractor([er.type for er in exophora_referents])

    @property
    def rels(self) -> List[str]:
        return ["="]

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.extractor.is_target(base_phrase):
                antecedent_candidates = self.extractor.get_candidates(base_phrase)
                referents = self.extractor.extract(base_phrase)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={"=": self._get_mention_tags(referents)},
                    )
                )
            else:
                cohesion_base_phrases.append(CohesionBasePhrase(base_phrase, is_target=False))
        return cohesion_base_phrases

    def _get_mention_tags(self, referents: List[Union[BasePhrase, ExophoraReferent]]) -> List[str]:
        mention_tags: List[str] = []
        for another_mention in referents:
            if isinstance(another_mention, BasePhrase):
                mention_tags.append(str(another_mention.global_index))
            else:
                assert isinstance(another_mention, ExophoraReferent)
                another_mention.index = None  # 不特定:人１ -> 不特定:人
                mention_tags.append(f"[{another_mention.text}]")  # 著者 -> [著者]
        return mention_tags or ["[NA]"]
