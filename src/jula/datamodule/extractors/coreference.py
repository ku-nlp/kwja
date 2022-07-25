import logging
from dataclasses import dataclass

from rhoknp import BasePhrase, Document
from rhoknp.rel import ExophoraReferent

from .base import Extractor, Phrase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreferenceAnnotation:
    arguments_set: list[list[str]]


class CoreferenceExtractor(Extractor):
    def __init__(
        self,
        exophors: list[ExophoraReferent],
        kc: bool = False,
    ) -> None:
        super().__init__(exophors, kc)

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> CoreferenceAnnotation:
        bp_list = document.base_phrases
        mentions_set: list[list[str]] = [[] for _ in bp_list]
        for sentence in document.sentences:
            for mention in sentence.base_phrases:
                is_target_phrase: bool = self.is_target(mention) and self._kc_skip_sentence(sentence, document) is False
                phrases[mention.global_index].is_target = is_target_phrase
                if is_target_phrase is False:
                    continue
                candidates: list[int] = [bp.global_index for bp in bp_list if self.is_candidate(bp, mention) is True]
                phrases[mention.global_index].candidates = candidates
                mentions_set[mention.global_index] = self._get_mentions(mention, candidates)

        return CoreferenceAnnotation(mentions_set)

    def _get_mentions(
        self,
        src_mention: BasePhrase,
        candidates: list[int],
    ) -> list[str]:
        if not src_mention.entities:
            return ["[NA]"]

        ment_strings: list[str] = []
        tgt_mentions = src_mention.get_coreferents(include_nonidentical=False)
        exophora_referents = [e.exophora_referent for e in src_mention.entities if e.exophora_referent is not None]
        for mention in tgt_mentions:
            if mention.global_index not in candidates:
                logger.debug(f"mention: {mention} in {mention.sentence.sid} is not in candidates and ignored")
                continue
            ment_strings.append(str(mention.global_index))
        for exophora_referent in exophora_referents:
            exophora_referent = self._relax_exophora_referent(exophora_referent)  # 不特定:人１ -> 不特定:人
            if exophora_referent in self.exophora_referents:
                ment_strings.append(str(exophora_referent))
        if ment_strings:
            return ment_strings
        else:
            return ["[NA]"]

    def is_target(self, bp: BasePhrase) -> bool:
        return self.is_coreference_target(bp)

    @staticmethod
    def is_coreference_target(bp: BasePhrase) -> bool:
        return bp.features.get("体言") is True

    @staticmethod
    def is_candidate(bp: BasePhrase, anaphor: BasePhrase) -> bool:
        return bp.global_index < anaphor.global_index
