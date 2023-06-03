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
from kwja.datamodule.examples import SpecialTokenIndexer, WordInferenceExample
from kwja.utils.cohesion_analysis import BridgingUtils, CohesionUtils, CoreferenceUtils, PasUtils
from kwja.utils.constants import SPLIT_INTO_WORDS_MODEL_NAMES, CohesionTask
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
        # ---------- cohesion analysis ----------
        self.cohesion_tasks = [CohesionTask(ct) for ct in cohesion_tasks]
        self.exophora_referents = [ExophoraReferent(er) for er in exophora_referents]
        self.pas_cases: List[str] = list(pas_cases)
        self.br_cases: List[str] = list(br_cases)
        self.cohesion_task2utils: Dict[CohesionTask, CohesionUtils] = {
            ct: self._build_cohesion_utils(ct, restrict_cohesion_target) for ct in self.cohesion_tasks
        }

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: List[str] = list(special_tokens)
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            add_special_tokens=False,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            is_split_into_words=True,
        ).encodings[0]

        self.examples: List[WordInferenceExample] = self._load_examples(self.doc_id2document)

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        tokenizer_input: Union[List[str], str] = [m.text for m in document_or_sentence.morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        return len(self.tokenizer.tokenize(tokenizer_input, is_split_into_words=self.tokenizer_input_format == "words"))

    def _load_examples(self, doc_id2document: Dict[str, Document]) -> List[WordInferenceExample]:
        examples = []
        example_id = 0
        for document in track(doc_id2document.values(), description="Loading examples"):
            tokenizer_input: Union[List[str], str] = [m.text for m in document.morphemes]
            if self.tokenizer_input_format == "text":
                tokenizer_input = " ".join(tokenizer_input)
            encoding: Encoding = self.tokenizer(
                tokenizer_input,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=False,
                is_split_into_words=self.tokenizer_input_format == "words",
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - len(self.special_tokens):
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

            examples.append(
                WordInferenceExample(
                    example_id=example_id,
                    encoding=merged_encoding,
                    special_token_indexer=special_token_indexer,
                    doc_id=document.doc_id,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: WordInferenceExample) -> WordModuleFeatures:
        document = self.doc_id2document[example.doc_id]

        # ---------- ner ----------
        # True/False = keep/mask
        ne_mask = [False] * self.max_seq_length
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                ne_mask[morpheme.global_index] = True

        # ---------- dependency parsing ----------
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        root_index = example.special_token_indexer.get_morpheme_level_index("[ROOT]")
        for sentence in document.sentences:
            for morpheme_src in sentence.morphemes:
                for morpheme_tgt in sentence.morphemes:
                    if morpheme_src.global_index != morpheme_tgt.global_index:
                        dependency_mask[morpheme_src.global_index][morpheme_tgt.global_index] = True
                dependency_mask[morpheme_src.global_index][root_index] = True

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
                for morpheme_global_index in example.special_token_indexer.get_morpheme_level_indices(
                    only_cohesion=True
                ):
                    rel_mask[morpheme.global_index][morpheme_global_index] = True
            cohesion_mask.extend([rel_mask] * len(cohesion_utils.rels))
        return WordModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.ids,
            attention_mask=example.encoding.attention_mask,
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
            ne_mask=ne_mask,
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
        word_ids: List[Union[int, None]],
        special_token_indexer: SpecialTokenIndexer,
        include_special_tokens: bool = True,
    ) -> List[List[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or token_index in special_token_indexer.token_level_indices:
                continue
            subword_map[word_id][token_index] = True
        if include_special_tokens is True:
            for token_index, morpheme_global_index in special_token_indexer.token_and_morpheme_level_indices:
                subword_map[morpheme_global_index][token_index] = True
        return subword_map

    def _build_cohesion_utils(self, cohesion_task: CohesionTask, restrict_cohesion_target: bool) -> CohesionUtils:
        if cohesion_task == CohesionTask.PAS_ANALYSIS:
            return PasUtils(self.pas_cases, "all", self.exophora_referents, restrict_cohesion_target)
        elif cohesion_task == CohesionTask.BRIDGING_REFERENCE_RESOLUTION:
            return BridgingUtils(self.br_cases, self.exophora_referents, restrict_cohesion_target)
        elif cohesion_task == CohesionTask.COREFERENCE_RESOLUTION:
            return CoreferenceUtils(self.exophora_referents, restrict_cohesion_target)
        else:
            raise ValueError("invalid cohesion task")
