import copy
import logging
from dataclasses import dataclass
from typing import Optional, Union

from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from rhoknp import BasePhrase
from rhoknp.cohesion import Argument, EndophoraArgument, ExophoraArgument, ExophoraReferent

from kwja.utils.sub_document import is_target_sentence

logger = logging.getLogger(__name__)


@dataclass
class CohesionBasePhrase:
    """A wrapper class of BasePhrase for cohesion analysis"""

    head_morpheme_global_index: int
    morpheme_global_indices: list[int]
    is_target: bool
    antecedent_candidates: Optional[list["CohesionBasePhrase"]] = None
    # case -> argument_tags / "=" -> mention_tags
    rel2tags: Optional[dict[str, list[str]]] = None


def wrap_base_phrases(
    base_phrases: list[BasePhrase], extractor: BaseExtractor, cases: list[str], restrict_cohesion_target: bool
) -> list[CohesionBasePhrase]:
    cohesion_base_phrases = [
        CohesionBasePhrase(
            base_phrase.head.global_index,
            [morpheme.global_index for morpheme in base_phrase.morphemes],
            is_target=False,
        )
        for base_phrase in base_phrases
    ]
    for base_phrase, cohesion_base_phrase in zip(base_phrases, cohesion_base_phrases):
        if is_target_sentence(base_phrase.sentence) and (
            restrict_cohesion_target is False or extractor.is_target(base_phrase)
        ):
            antecedent_candidates = extractor.get_candidates(base_phrase, base_phrase.document.base_phrases)
            all_rels = extractor.extract_rels(base_phrase)
            if isinstance(extractor, (PasExtractor, BridgingExtractor)):
                assert isinstance(all_rels, dict)
                rel2tags: dict[str, list[str]] = {case: _get_argument_tags(all_rels[case]) for case in cases}
            elif isinstance(extractor, CoreferenceExtractor):
                assert isinstance(all_rels, list)
                rel2tags = {"=": _get_referent_tags(all_rels)}
            else:
                raise AssertionError
            cohesion_base_phrase.is_target = True
            cohesion_base_phrase.antecedent_candidates = [
                cohesion_base_phrases[cand.global_index] for cand in antecedent_candidates
            ]
            cohesion_base_phrase.rel2tags = rel2tags
    return cohesion_base_phrases


def _get_argument_tags(arguments: list[Argument]) -> list[str]:
    """Get argument tags.

    Note:
        endophora argument: string of base phrase global index
        exophora argument: exophora referent
        no argument: [NULL]
    """
    argument_tags: list[str] = []
    for argument in arguments:
        if isinstance(argument, EndophoraArgument):
            argument_tag = str(argument.base_phrase.global_index)
        else:
            assert isinstance(argument, ExophoraArgument)
            exophora_referent = copy.copy(argument.exophora_referent)
            exophora_referent.index = None  # 不特定:人１ -> 不特定:人
            argument_tag = f"[{exophora_referent.text}]"  # 不特定:人 -> [不特定:人]
        argument_tags.append(argument_tag)
    return argument_tags or ["[NULL]"]


def _get_referent_tags(referents: list[Union[BasePhrase, ExophoraReferent]]) -> list[str]:
    """Get referent tags.

    Note:
        endophora referent: string of base phrase global index
        exophora referent: exophora referent text wrapped by []
        no referent: [NA]
    """
    mention_tags: list[str] = []
    for referent in referents:
        if isinstance(referent, BasePhrase):
            mention_tag = str(referent.global_index)
        else:
            assert isinstance(referent, ExophoraReferent)
            referent.index = None  # 不特定:人１ -> 不特定:人
            mention_tag = f"[{referent.text}]"  # 著者 -> [著者]
        mention_tags.append(mention_tag)
    return mention_tags or ["[NA]"]
