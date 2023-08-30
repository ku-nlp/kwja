import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from rhoknp import BasePhrase
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
        if attr_name in {"base_phrase", "is_target", "antecedent_candidates", "rel2tags"}:
            return getattr(self, attr_name) if attr_name in self.__dict__ else None
        else:
            return getattr(self.base_phrase, attr_name)

    def __getstate__(self) -> Dict[str, Any]:
        # the attributes are not set after unpickling unless this method is defined, which causes an error when num_workers > 0.
        return {
            "base_phrase": self.base_phrase,
            "is_target": self.is_target,
            "antecedent_candidates": self.antecedent_candidates,
            "rel2tags": self.rel2tags,
        }


def wrap_base_phrase(
    base_phrase: BasePhrase, extractor: BaseExtractor, cases: List[str], restrict_cohesion_target: bool
) -> CohesionBasePhrase:
    if is_target_sentence(base_phrase.sentence) and (
        restrict_cohesion_target is False or extractor.is_target(base_phrase)
    ):
        antecedent_candidates = extractor.get_candidates(base_phrase, base_phrase.document.base_phrases)
        all_rels = extractor.extract_rels(base_phrase)
        if isinstance(extractor, (PasExtractor, BridgingExtractor)):
            rel2tags: Dict[str, List[str]] = {case: _get_argument_tags(all_rels[case]) for case in cases}
        else:
            assert isinstance(extractor, CoreferenceExtractor)
            rel2tags = {"=": _get_referent_tags(all_rels)}
        return CohesionBasePhrase(
            base_phrase,
            is_target=True,
            antecedent_candidates=antecedent_candidates,
            rel2tags=rel2tags,
        )
    else:
        return CohesionBasePhrase(base_phrase, is_target=False)


def _get_argument_tags(arguments: List[Argument]) -> List[str]:
    """Get argument tags.

    Note:
        endophora argument: string of base phrase global index
        exophora argument: exophora referent
        no argument: [NULL]
    """
    argument_tags: List[str] = []
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


def _get_referent_tags(referents: List[Union[BasePhrase, ExophoraReferent]]) -> List[str]:
    """Get referent tags.

    Note:
        endophora referent: string of base phrase global index
        exophora referent: exophora referent text wrapped by []
        no referent: [NA]
    """
    mention_tags: List[str] = []
    for referent in referents:
        if isinstance(referent, BasePhrase):
            mention_tag = str(referent.global_index)
        else:
            assert isinstance(referent, ExophoraReferent)
            referent.index = None  # 不特定:人１ -> 不特定:人
            mention_tag = f"[{referent.text}]"  # 著者 -> [著者]
        mention_tags.append(mention_tag)
    return mention_tags or ["[NA]"]
