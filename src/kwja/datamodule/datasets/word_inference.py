import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent
from rhoknp.utils.reader import chunk_by_document
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.datasets.word import WordModuleFeatures
from kwja.datamodule.examples import WordInferenceExample
from kwja.utils.cohesion_analysis import BridgingUtils, CohesionUtils, CoreferenceUtils, PasUtils
from kwja.utils.constants import SPLIT_INTO_WORDS_MODEL_NAMES, CohesionTask
from kwja.utils.progress_bar import track
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
        juman_file: Optional[Path] = None,
        knp_file: Optional[Path] = None,
    ) -> None:
        super(WordInferenceDataset, self).__init__(tokenizer, max_seq_length)
        if juman_file is not None:
            with juman_file.open() as f:
                documents = [
                    Document.from_jumanpp(c) for c in track(chunk_by_document(f), description="Loading documents")
                ]
        elif knp_file is not None:
            with knp_file.open() as f:
                documents = [Document.from_knp(c) for c in track(chunk_by_document(f), description="Loading documents")]
        else:
            # do_predict_after_train
            documents = []

        if tokenizer.name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
            self.tokenizer_input_format: Literal["words", "text"] = "words"
        else:
            self.tokenizer_input_format = "text"

        super(BaseDataset, self).__init__(documents, tokenizer, max_seq_length, document_split_stride)
        # ---------- seq2seq ----------
        self.from_seq2seq: bool = juman_file is not None and juman_file.suffix == ".seq2seq"

        # ---------- cohesion analysis ----------
        self.cohesion_tasks = [CohesionTask(t) for t in cohesion_tasks]
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.pas_cases: List[str] = list(pas_cases)
        self.br_cases: List[str] = list(br_cases)
        self.cohesion_task2utils: Dict[CohesionTask, CohesionUtils] = {}
        for cohesion_task in self.cohesion_tasks:
            if cohesion_task == CohesionTask.PAS_ANALYSIS:
                self.cohesion_task2utils[cohesion_task] = PasUtils(
                    self.pas_cases, "all", self.exophora_referents, restrict_target=restrict_cohesion_target
                )
            elif cohesion_task == CohesionTask.BRIDGING_REFERENCE_RESOLUTION:
                self.cohesion_task2utils[cohesion_task] = BridgingUtils(
                    self.br_cases, self.exophora_referents, restrict_target=restrict_cohesion_target
                )
            elif cohesion_task == CohesionTask.COREFERENCE_RESOLUTION:
                self.cohesion_task2utils[cohesion_task] = CoreferenceUtils(
                    self.exophora_referents, restrict_target=restrict_cohesion_target
                )

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: List[str] = list(special_tokens)
        self.special_token2index: Dict[str, int] = {
            st: self.max_seq_length - len(self.special_tokens) + i for i, st in enumerate(self.special_tokens)
        }
        self.index2special_token: Dict[int, str] = {v: k for k, v in self.special_token2index.items()}
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        self.examples: List[WordInferenceExample] = self._load_examples(self.documents)

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        tokenizer_input: Union[List[str], str] = [m.text for m in document_or_sentence.morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        encoding = self.tokenizer(
            tokenizer_input, add_special_tokens=False, is_split_into_words=self.tokenizer_input_format == "words"
        ).encodings[0]
        return len(encoding.ids)

    def _load_examples(self, documents: List[Document]) -> List[WordInferenceExample]:
        examples = []
        example_id = 0
        for document in track(documents, description="Loading examples"):
            tokenizer_input: Union[List[str], str] = [m.text for m in document.morphemes]
            if self.tokenizer_input_format == "text":
                tokenizer_input = " ".join(tokenizer_input)
            encoding: Encoding = self.tokenizer(
                tokenizer_input,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
                is_split_into_words=self.tokenizer_input_format == "words",
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue
            num_tokenized_morphemes = len({word_id for word_id in encoding.word_ids if word_id is not None})
            if len(document.morphemes) != num_tokenized_morphemes:
                logger.warning(f"Document length and tokenized length mismatch: {document.text}")
                continue

            examples.append(
                WordInferenceExample(
                    example_id=example_id,
                    encoding=encoding,
                    doc_id=document.doc_id,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: WordInferenceExample) -> WordModuleFeatures:
        document = self.doc_id2document[example.doc_id]

        target_mask = [0 for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = 1

        # ---------- dependency parsing ----------
        # True/False = keep/mask
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for sentence in document.sentences:
            morpheme_global_indices = [morpheme.global_index for morpheme in sentence.morphemes]
            start, stop = min(morpheme_global_indices), max(morpheme_global_indices) + 1
            for i in range(start, stop):
                for j in range(start, stop):
                    if i != j:
                        dependency_mask[i][j] = True
                dependency_mask[i][self.special_token2index["[ROOT]"]] = True

        # ---------- cohesion analysis ----------
        cohesion_mask: List[List[List[bool]]] = []  # (rel, seq, seq)
        morphemes = document.morphemes
        for cohesion_task, cohesion_utils in self.cohesion_task2utils.items():
            rel_mask: List[List[bool]] = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
            for morpheme in morphemes:
                for antecedent_candidate_morpheme in cohesion_utils.get_antecedent_candidate_morphemes(
                    morpheme, morphemes
                ):
                    rel_mask[morpheme.global_index][antecedent_candidate_morpheme.global_index] = True
                for cohesion_special_index in self.cohesion_special_indices:
                    rel_mask[morpheme.global_index][cohesion_special_index] = True
            cohesion_mask.extend([rel_mask] * len(cohesion_utils.rels))

        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])

        return WordModuleFeatures(
            example_ids=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            target_mask=target_mask,
            subword_map=self._get_subword_map(merged_encoding),
            reading_labels=[],
            reading_subword_map=self._get_subword_map(merged_encoding, include_special_tokens=False),
            pos_labels=[],
            subpos_labels=[],
            conjtype_labels=[],
            conjform_labels=[],
            word_feature_labels=[],
            ne_labels=[],
            base_phrase_feature_labels=[],
            dependency_labels=[],
            dependency_mask=dependency_mask,
            dependency_type_labels=[],
            cohesion_labels=[],
            cohesion_mask=cohesion_mask,
            discourse_labels=[],
        )

    def _get_subword_map(self, encoding: Encoding, include_special_tokens: bool = True) -> List[List[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is None or token_id in self.special_indices:
                continue
            subword_map[word_id][token_id] = True
        if include_special_tokens:
            for special_index in self.special_indices:
                subword_map[special_index][special_index] = True
        return subword_map

    @property
    def cohesion_special_indices(self) -> List[int]:
        return [i for st, i in self.special_token2index.items() if st != "[ROOT]"]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def special_indices(self) -> List[int]:
        return list(self.special_token2index.values())
