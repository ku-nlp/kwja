import logging
from dataclasses import dataclass

from kyoto_reader import BasePhrase, Document

from .base import Extractor, Phrase

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class CoreferenceAnnotation:
    arguments_set: list[list[str]]


class CoreferenceExtractor(Extractor):
    def __init__(
        self,
        # corefs: list[str],
        exophors: list[str],
        kc: bool = False,
    ) -> None:
        super().__init__(exophors, kc)
        # self.corefs = corefs

    def extract(
        self,
        document: Document,
        phrases: list[Phrase],
    ) -> CoreferenceAnnotation:
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
                arguments_set[anaphor.dtid] = self._get_mentions(
                    anaphor, document, candidates
                )

        return CoreferenceAnnotation(arguments_set)

    def _get_mentions(
        self,
        src_bp: BasePhrase,
        document: Document,
        candidates: list[int],
    ) -> list[str]:
        if src_bp.dtid in document.mentions:
            ment_strings: list[str] = []
            src_mention = document.mentions[src_bp.dtid]
            tgt_mentions = document.get_siblings(
                src_mention, relax=False
            )  # exclude uncertain entities
            exophors = [
                document.entities[eid].exophor
                for eid in src_mention.eids
                if document.entities[eid].is_special
            ]
            for mention in tgt_mentions:
                if mention.dtid not in candidates:
                    logger.debug(
                        f"mention: {mention} in {document.doc_id} is not in candidates and ignored"
                    )
                    continue
                ment_strings.append(str(mention.dtid))
            for exophor in exophors:
                if exophor in self.relax_exophors:
                    ment_strings.append(self.relax_exophors[exophor])  # 不特定:人１ -> 不特定:人
            if ment_strings:
                return ment_strings
            else:
                return ["NA"]  # force cataphor to point [NA]
        else:
            return ["NA"]

    def is_target(self, bp: BasePhrase) -> bool:
        return self.is_coreference_target(bp)

    @staticmethod
    def is_coreference_target(bp: BasePhrase) -> bool:
        return "体言" in bp.tag.features

    @staticmethod
    def is_candidate(bp: BasePhrase, anaphor: BasePhrase) -> bool:
        return bp.dtid < anaphor.dtid
