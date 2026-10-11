import logging
from pathlib import Path

import torch
from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from omegaconf import ListConfig
from rhoknp import Document, Morpheme, Sentence
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from rhoknp.utils.reader import chunk_by_document
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.datasets.word import WordModuleFeatures
from kwja.datamodule.examples import SpecialTokenIndexer, WordInferenceExample
from kwja.utils.constants import CohesionTask
from kwja.utils.logging_util import track
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


class WordInferenceDataset(BaseDataset[WordInferenceExample, WordModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        document_split_stride: int,
        cohesion_tasks: ListConfig,
        exophora_referents: ListConfig,
        restrict_cohesion_target: bool,
        pas_cases: ListConfig,
        br_cases: ListConfig,
        special_tokens: ListConfig,
        juman_file: Path | None = None,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)
        if juman_file is not None:
            with juman_file.open(encoding="utf-8") as f:
                documents = [
                    Document.from_jumanpp(c) for c in track(chunk_by_document(f), description="Loading documents")
                ]
        else:
            # do_predict_after_train
            documents = []

        super(BaseDataset, self).__init__(documents, tokenizer, max_seq_length, document_split_stride)
        # ---------- cohesion analysis ----------
        self.cohesion_tasks: list[CohesionTask] = [task for task in CohesionTask if task.value in cohesion_tasks]
        self.exophora_referent_types: list[ExophoraReferentType] = [
            ExophoraReferent(er).type for er in exophora_referents
        ]
        self.cohesion_task2extractor: dict[CohesionTask, BaseExtractor] = {
            CohesionTask.PAS_ANALYSIS: PasExtractor(
                list(pas_cases),
                self.exophora_referent_types,
                verbal_predicate=True,
                nominal_predicate=True,
            ),
            CohesionTask.BRIDGING_REFERENCE_RESOLUTION: BridgingExtractor(list(br_cases), self.exophora_referent_types),
            CohesionTask.COREFERENCE_RESOLUTION: CoreferenceExtractor(self.exophora_referent_types),
        }
        self.cohesion_task2rels: dict[CohesionTask, list[str]] = {
            CohesionTask.PAS_ANALYSIS: list(pas_cases),
            CohesionTask.BRIDGING_REFERENCE_RESOLUTION: list(br_cases),
            CohesionTask.COREFERENCE_RESOLUTION: ["="],
        }
        self.restrict_cohesion_target: bool = restrict_cohesion_target

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: list[str] = [st for st in special_tokens if st != " "]
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            add_special_tokens=False,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            is_split_into_words=True,
        ).encodings[0]

        # sub documents that do not fit in max_seq_length are left unanalyzed; kept so that callers can report them
        self.skipped_documents: list[Document] = []
        self.examples: list[WordInferenceExample] = self._load_examples(self.doc_id2document)

    def _get_tokenized_len(self, document_or_sentence: Document | Sentence) -> int:
        tokenizer_input: list[str] = [m.text for m in document_or_sentence.morphemes]
        return len(self.tokenizer(tokenizer_input, add_special_tokens=False, is_split_into_words=True).tokens())

    def _load_examples(self, doc_id2document: dict[str, Document]) -> list[WordInferenceExample]:
        examples = []
        example_id = 0
        for document in track(doc_id2document.values(), description="Loading examples"):
            tokenizer_input: list[str] | str = [m.text for m in document.morphemes]
            encoding: Encoding = self.tokenizer(
                tokenizer_input,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=False,
                is_split_into_words=True,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - len(self.special_tokens):
                logger.warning(f"Length of sub document is too long: {document.text}")
                self.skipped_documents.append(document)
                continue
            padding_encoding: Encoding = self.tokenizer(
                "",
                add_special_tokens=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - len(encoding.ids) - len(self.special_tokens),
            ).encodings[0]
            merged_encoding: Encoding = Encoding.merge([encoding, self.special_encoding, padding_encoding])

            special_token_indexer = SpecialTokenIndexer(self.special_tokens, len(encoding.ids), len(document.morphemes))

            analysis_target_morpheme_indices = []
            for sentence in extract_target_sentences(document):
                analysis_target_morpheme_indices += [m.global_index for m in sentence.morphemes]

            examples.append(
                WordInferenceExample(
                    example_id=example_id,
                    encoding=merged_encoding,
                    special_token_indexer=special_token_indexer,
                    doc_id=document.doc_id,
                    analysis_target_morpheme_indices=analysis_target_morpheme_indices,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: WordInferenceExample) -> WordModuleFeatures:
        document = self.doc_id2document[example.doc_id]

        # ---------- ner ----------
        target_mask = [False] * self.max_seq_length
        for global_index in example.analysis_target_morpheme_indices:
            target_mask[global_index] = True

        # ---------- dependency parsing ----------
        # Build the O(seq^2) masks as tensors instead of nested Python lists: building
        # max_seq_length**2 (x #rels) bool lists and converting them with torch.as_tensor
        # in the collator used to dominate the collation time.
        dependency_mask = torch.zeros((self.max_seq_length, self.max_seq_length), dtype=torch.bool)
        root_index = example.special_token_indexer.get_morpheme_level_index("[ROOT]")
        for sentence in document.sentences:
            sentence_morphemes = sentence.morphemes
            if not sentence_morphemes:
                continue
            # Morphemes in a sentence have contiguous global indices.
            start = sentence_morphemes[0].global_index
            stop = sentence_morphemes[-1].global_index + 1
            dependency_mask[start:stop, start:stop] = True
            diagonal = torch.arange(start, stop)
            dependency_mask[diagonal, diagonal] = False
            dependency_mask[start:stop, root_index] = True

        # ---------- cohesion analysis ----------
        # Asking the extractor about every morpheme pair costs O(seq^2) Python calls and
        # dominates the dataset time for long documents. The built-in extractors decide
        # from the global indices and the sentences of the two morphemes alone, so the
        # whole block is computed at once; any other extractor keeps the generic path.
        rel_masks: list[torch.Tensor] = []
        morphemes = document.morphemes
        special_indices = example.special_token_indexer.get_morpheme_level_indices(only_cohesion=True)
        num_morphemes = len(morphemes)
        global_indices = torch.arange(num_morphemes)
        sentence_indices = _sentence_indices(morphemes)
        for cohesion_task in self.cohesion_tasks:
            cohesion_rels = self.cohesion_task2rels[cohesion_task]
            cohesion_extractor = self.cohesion_task2extractor[cohesion_task]
            rel_mask = torch.zeros((self.max_seq_length, self.max_seq_length), dtype=torch.bool)
            candidate_block = _candidate_block(cohesion_extractor, global_indices, sentence_indices)
            if candidate_block is None:
                for morpheme in morphemes:
                    candidate_indices = [c.global_index for c in cohesion_extractor.get_candidates(morpheme, morphemes)]
                    if candidate_indices:
                        rel_mask[morpheme.global_index, candidate_indices] = True
                    if special_indices:
                        rel_mask[morpheme.global_index, special_indices] = True
            else:
                rel_mask[:num_morphemes, :num_morphemes] = candidate_block
                if special_indices:
                    rel_mask[:num_morphemes, special_indices] = True
            rel_masks.extend(rel_mask for _ in cohesion_rels)
        cohesion_mask = torch.stack(rel_masks)  # (rel, seq, seq)
        return WordModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.ids,
            attention_mask=example.encoding.attention_mask,
            special_token_indices=example.special_token_indexer.token_level_indices,
            subword_map=self._generate_subword_map(example.encoding.word_ids, example.special_token_indexer),
            reading_labels=[],
            reading_subword_map=self._generate_subword_map(
                example.encoding.word_ids, example.special_token_indexer, include_special_tokens=False
            ),
            pos_labels=[],
            subpos_labels=[],
            conjtype_labels=[],
            conjform_labels=[],
            word_feature_labels=[],
            ne_labels=[],
            ne_mask=target_mask,
            base_phrase_feature_labels=[],
            dependency_labels=[],
            dependency_mask=dependency_mask,
            dependency_type_labels=[],
            cohesion_labels=[],
            cohesion_mask=cohesion_mask,
            discourse_labels=[],
        )

    def _generate_subword_map(
        self,
        word_ids: list[int | None],
        special_token_indexer: SpecialTokenIndexer,
        include_special_tokens: bool = True,
    ) -> torch.Tensor:
        subword_map = torch.zeros((self.max_seq_length, self.max_seq_length), dtype=torch.bool)
        special_token_level_indices = set(special_token_indexer.token_level_indices)
        rows: list[int] = []
        cols: list[int] = []
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or token_index in special_token_level_indices:
                continue
            rows.append(word_id)
            cols.append(token_index)
        if include_special_tokens is True:
            for token_index, morpheme_global_index in special_token_indexer.token_and_morpheme_level_indices:
                rows.append(morpheme_global_index)
                cols.append(token_index)
        if rows:
            subword_map[rows, cols] = True
        return subword_map


def _sentence_indices(morphemes: list[Morpheme]) -> torch.Tensor:
    """Number the sentences the morphemes belong to, in order of appearance."""
    sid2index: dict[str, int] = {}
    return torch.tensor([sid2index.setdefault(m.sentence.sid, len(sid2index)) for m in morphemes], dtype=torch.long)


def _candidate_block(
    cohesion_extractor: BaseExtractor, global_indices: torch.Tensor, sentence_indices: torch.Tensor
) -> torch.Tensor | None:
    """Return `is_candidate` for every (anaphor, candidate) pair, or None if the rule is unknown.

    PasExtractor and BridgingExtractor take every preceding morpheme plus the following
    ones of the same sentence, CoreferenceExtractor the preceding ones only. Extractors
    that override the rule fall back to calling `get_candidates` morpheme by morpheme.
    """
    is_anaphora = global_indices[None, :] < global_indices[:, None]
    if type(cohesion_extractor) is CoreferenceExtractor:
        return is_anaphora
    if type(cohesion_extractor) in (PasExtractor, BridgingExtractor):
        is_intra_sentential_cataphora = (global_indices[None, :] > global_indices[:, None]) & (
            sentence_indices[None, :] == sentence_indices[:, None]
        )
        return is_anaphora | is_intra_sentential_cataphora
    return None
