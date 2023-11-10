import logging
import sys
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Generic, List, TypeVar, Union

from rhoknp import Document, Sentence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from kwja.utils.logging_util import track
from kwja.utils.sub_document import SequenceSplitter, SpanCandidate, to_sub_doc_id

sys.setrecursionlimit(5000)

logger = logging.getLogger(__name__)

ExampleType = TypeVar("ExampleType")
FeatureType = TypeVar("FeatureType")


class BaseDataset(Dataset[FeatureType], Generic[ExampleType, FeatureType], ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_seq_length: int = max_seq_length
        self.examples: List[ExampleType] = []

    def __getitem__(self, index) -> FeatureType:
        return self.encode(self.examples[index])

    def __len__(self) -> int:
        return len(self.examples)

    def encode(self, example: ExampleType) -> FeatureType:
        raise NotImplementedError


class FullAnnotatedDocumentLoaderMixin:
    def __init__(
        self,
        source: Union[Path, List[Document]],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        document_split_stride: int,
        ext: str = "knp",
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        orig_documents: List[Document]
        if isinstance(source, Path):
            assert source.is_dir()
            orig_documents = self._load_documents(source, ext)
        else:
            orig_documents = source

        self.doc_id2document: Dict[str, Document] = {}
        for orig_document in track(orig_documents, description="Splitting documents"):
            orig_document = self._postprocess_document(orig_document)
            self.doc_id2document.update(
                {
                    document.doc_id: document
                    for document in self._split_document(
                        orig_document,
                        max_seq_length - len(self.tokenizer.additional_special_tokens) - 2,  # -2: [CLS] and [SEP]
                        stride=document_split_stride,
                    )
                }
            )

    @staticmethod
    def _load_documents(document_dir: Path, ext: str) -> List[Document]:
        documents = []
        # Use a ProcessPoolExecutor to parallelize the loading of documents
        with ProcessPoolExecutor(4) as executor:
            paths = sorted(document_dir.glob(f"*.{ext}"))
            for document in track(
                executor.map(FullAnnotatedDocumentLoaderMixin._load_document, paths),
                total=len(paths),
                description="Loading documents",
            ):
                documents.append(document)
        return documents

    @staticmethod
    def _load_document(path: Path) -> Document:
        return Document.from_knp(path.read_text())

    def _postprocess_document(self, document: Document) -> Document:
        return document

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> List[Document]:
        sentence_tokens = [self._get_tokenized_len(sentence) for sentence in document.sentences]
        if sum(sentence_tokens) <= max_token_length:
            return [document]

        splitter = SequenceSplitter(sentence_tokens, max_token_length, stride)
        sub_documents: List[Document] = []
        sub_idx = 0
        for span in splitter.split_into_spans():
            assert isinstance(span, SpanCandidate)
            sentences = document.sentences[span.start : span.end]
            sub_document = Document.from_sentences(sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=span.stride)
            sub_document.doc_id = sub_doc_id
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.comment = sentence.comment
            sub_documents.append(sub_document)
            sub_idx += 1
        return sub_documents

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(document_or_sentence.text))
