import logging
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Union

from rhoknp import Document, Sentence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.progress_bar import track
from kwja.utils.sub_document import SequenceSplitter, SpanCandidate, to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        source: Union[Path, str, List[Document]],
        document_split_stride: int,
        model_name_or_path: str,
        max_seq_length: int,
        tokenizer_kwargs: dict,
        ext: str = "knp",
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length
        self.orig_documents: List[Document]
        if isinstance(source, (Path, str)):
            source = Path(source)
            assert source.is_dir()
            self.orig_documents = self._load_documents(source, ext)
        else:
            self.orig_documents = source
        self.doc_id2document: Dict[str, Document] = {}
        for orig_document in track(self.orig_documents, description="Splitting documents"):
            orig_document = self._normalize(orig_document)
            self.doc_id2document.update(
                {
                    document.doc_id: document
                    for document in self._split_document(
                        orig_document,
                        self.max_seq_length - len(self.tokenizer.additional_special_tokens) - 2,  # -2: [CLS] and [SEP]
                        stride=document_split_stride,
                    )
                }
            )

    @cached_property
    def documents(self) -> List[Document]:
        return list(self.doc_id2document.values())

    @staticmethod
    def _load_documents(document_dir: Path, ext: str = "knp") -> List[Document]:
        documents = []
        for path in track(sorted(document_dir.glob(f"*.{ext}")), description="Loading documents"):
            # TODO: fix document files that raise exception
            try:
                documents.append(Document.from_knp(path.read_text()))
            except AssertionError:
                logger.warning(f"{path} is not a valid knp file.")
        return documents

    def _normalize(self, document: Document) -> Document:
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
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.doc_id = sub_doc_id
                sub_sentence.sid = sentence.sid
                sub_sentence.misc_comment = sentence.misc_comment
            sub_document.doc_id = sub_doc_id
            sub_documents.append(sub_document)
            sub_idx += 1
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))
