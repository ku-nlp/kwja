import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
    def __init__(
        self,
        exophora_referents: List[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        self.exophora_referents = exophora_referents
        self.restrict_target = restrict_target

    @property
    def rels(self) -> List[str]:
        raise NotImplementedError

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        raise NotImplementedError

    def is_target(self, base_phrase: BasePhrase) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_antecedent_candidate(antecedent: Union[BasePhrase, Morpheme], anaphor: Union[BasePhrase, Morpheme]) -> bool:
        raise NotImplementedError

    def get_antecedent_candidates(self, base_phrase: BasePhrase, base_phrases: List[BasePhrase]) -> List[BasePhrase]:
        return [bp for bp in base_phrases if self.is_antecedent_candidate(bp, base_phrase)]

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
        super().__init__(exophora_referents, restrict_target)
        self.cases = cases
        self.target = target

    @property
    def rels(self) -> List[str]:
        return self.cases

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases: List[CohesionBasePhrase] = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.is_target(base_phrase):
                antecedent_candidates = self.get_antecedent_candidates(base_phrase, base_phrases)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={
                            case: self._get_argument_tags(
                                base_phrase.pas.get_arguments(case, relax=False), antecedent_candidates
                            )
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
                if argument.exophora_referent in self.exophora_referents:
                    target_arguments.append(argument)
                elif argument.exophora_referent.text == "[不明]":
                    return []  # don't train uncertain argument

        argument_tags: List[str] = []
        for target_argument in target_arguments:
            if isinstance(target_argument, EndophoraArgument):
                argument_tag = str(target_argument.base_phrase.global_index)
            else:
                assert isinstance(target_argument, ExophoraArgument)
                argument_tag = f"[{target_argument.exophora_referent.text}]"  # 著者 -> [著者]
            argument_tags.append(argument_tag)
        return argument_tags or ["[NULL]"]

    def is_target(self, base_phrase: BasePhrase) -> bool:
        verbal = self.target in ("pred", "all")
        nominal = self.target in ("noun", "all")
        return (self.restrict_target is False) or is_pas_target(base_phrase, verbal, nominal)

    @staticmethod
    def is_antecedent_candidate(antecedent: Union[BasePhrase, Morpheme], anaphor: Union[BasePhrase, Morpheme]) -> bool:
        anaphora = antecedent.global_index < anaphor.global_index
        # 文内の後方照応は許す
        cataphora = (antecedent.global_index > anaphor.global_index) and (
            antecedent.sentence.sid == anaphor.sentence.sid
        )
        return anaphora or cataphora


class BridgingUtils(CohesionUtils):
    def __init__(
        self,
        cases: List[str],
        exophora_referents: List[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        super().__init__(exophora_referents, restrict_target)
        assert "ノ" in cases, '"ノ" case isn\'t found'
        self.cases = cases

    @property
    def rels(self) -> List[str]:
        return ["ノ"]

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases: List[CohesionBasePhrase] = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.is_target(base_phrase):
                arguments = []
                for case in self.cases:
                    arguments += base_phrase.pas.get_arguments(case, relax=False)
                antecedent_candidates = self.get_antecedent_candidates(base_phrase, base_phrases)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={"ノ": self._get_argument_tags(arguments, antecedent_candidates)},
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
                if argument.exophora_referent in self.exophora_referents:
                    target_arguments.append(argument)
                elif argument.exophora_referent.text == "[不明]":
                    return []  # don't train uncertain argument

        argument_tags: List[str] = []
        for target_argument in target_arguments:
            if isinstance(target_argument, EndophoraArgument):
                argument_tag = str(target_argument.base_phrase.global_index)
            else:
                assert isinstance(target_argument, ExophoraArgument)
                argument_tag = f"[{target_argument.exophora_referent.text}]"  # 著者 -> [著者]
            argument_tags.append(argument_tag)
        return argument_tags or ["[NULL]"]

    def is_target(self, base_phrase: BasePhrase) -> bool:
        return (self.restrict_target is False) or is_bridging_target(base_phrase)

    @staticmethod
    def is_antecedent_candidate(antecedent: Union[BasePhrase, Morpheme], anaphor: Union[BasePhrase, Morpheme]) -> bool:
        anaphora = antecedent.global_index < anaphor.global_index
        # referents of intra-sentential cataphora are included in the candidates
        cataphora = (antecedent.global_index > anaphor.global_index) and (
            antecedent.sentence.sid == anaphor.sentence.sid
        )
        return anaphora or cataphora


class CoreferenceUtils(CohesionUtils):
    def __init__(
        self,
        exophora_referents: List[ExophoraReferent],
        restrict_target: bool,
    ) -> None:
        super().__init__(exophora_referents, restrict_target)

    @property
    def rels(self) -> List[str]:
        return ["="]

    def wrap(self, base_phrases: List[BasePhrase]) -> List[CohesionBasePhrase]:
        cohesion_base_phrases = []
        for base_phrase in base_phrases:
            if is_target_sentence(base_phrase.sentence) and self.is_target(base_phrase):
                antecedent_candidates = self.get_antecedent_candidates(base_phrase, base_phrases)
                cohesion_base_phrases.append(
                    CohesionBasePhrase(
                        base_phrase,
                        is_target=True,
                        antecedent_candidates=antecedent_candidates,
                        rel2tags={"=": self._get_mention_tags(base_phrase, antecedent_candidates)},
                    )
                )
            else:
                cohesion_base_phrases.append(CohesionBasePhrase(base_phrase, is_target=False))
        return cohesion_base_phrases

    def _get_mention_tags(self, mention: BasePhrase, antecedent_candidates: List[BasePhrase]) -> List[str]:
        mention_tags: List[str] = []
        for another_mention in mention.get_coreferents(include_nonidentical=False):
            if another_mention in antecedent_candidates:
                mention_tags.append(str(another_mention.global_index))
            else:
                logger.info(f'mention "{another_mention}" is ignored ({another_mention.sentence.sent_id})')

        for exophora_referent in [e.exophora_referent for e in mention.entities if e.exophora_referent is not None]:
            exophora_referent.index = None  # 不特定:人１ -> 不特定:人
            if exophora_referent in self.exophora_referents:
                mention_tags.append(f"[{exophora_referent.text}]")  # 著者 -> [著者]
        return mention_tags or ["[NA]"]

    def is_target(self, base_phrase: BasePhrase) -> bool:
        return (self.restrict_target is False) or is_coreference_target(base_phrase)

    @staticmethod
    def is_antecedent_candidate(antecedent: Union[BasePhrase, Morpheme], anaphor: Union[BasePhrase, Morpheme]) -> bool:
        anaphora = antecedent.global_index < anaphor.global_index
        return anaphora


def is_pas_target(base_phrase: BasePhrase, verbal: bool, nominal: bool) -> bool:
    return (verbal and "用言" in base_phrase.features) or (nominal and "非用言格解析" in base_phrase.features)


def is_bridging_target(base_phrase: BasePhrase):
    return (base_phrase.features.get("体言") is True) and ("非用言格解析" not in base_phrase.features)


def is_coreference_target(base_phrase: BasePhrase) -> bool:
    return base_phrase.features.get("体言") is True
