import logging
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

from rhoknp import Document, Sentence
from rhoknp.utils.reader import chunk_by_document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.sub_document import to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseInferenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        document_split_stride: int,
        model_name_or_path: str,
        max_seq_length: int,
        tokenizer_kwargs: dict,
        doc_id_prefix: Optional[str] = None,
        juman_file: Optional[Path] = None,
        knp_file: Optional[Path] = None,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length
        if knp_file is not None:
            with knp_file.open(mode="r") as f:
                orig_documents = [Document.from_knp(c) for c in chunk_by_document(f)]
        elif juman_file is not None:
            with juman_file.open(mode="r") as f:
                orig_documents = [Document.from_jumanpp(c) for c in chunk_by_document(f)]
        else:
            orig_documents = self._create_documents_from_texts(list(texts), doc_id_prefix)
        self.orig_documents: List[Document] = orig_documents
        self.doc_id2document: Dict[str, Document] = {}
        for orig_document in self.orig_documents:
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
    def _create_documents_from_texts(texts: List[str], doc_id_prefix: Optional[str]) -> List[Document]:
        raise NotImplementedError

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> List[Document]:
        cum_lens = [0]
        for sentence in document.sentences:
            num_tokens = self._get_tokenized_len(sentence)
            cum_lens.append(cum_lens[-1] + num_tokens)
        if cum_lens[-1] <= max_token_length:
            return [document]

        end = 1
        # end を探索
        while end < len(document.sentences) and cum_lens[end + 1] - cum_lens[0] <= max_token_length:
            end += 1

        sub_documents: List[Document] = []
        sub_idx = 0
        while end <= len(document.sentences):
            start = 0
            # start を探索
            while cum_lens[end] - cum_lens[start] > max_token_length:
                if start == end - 1:
                    break
                start += 1

            sub_sentences = document.sentences[start:end]
            sub_document = Document.from_sentences(sub_sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=stride)
            for sentence, sub_sentence in zip(sub_document.sentences, sub_sentences):
                sentence.doc_id = sub_doc_id
                sentence.sid = sub_sentence.sid
            sub_document.doc_id = sub_doc_id
            sub_documents.append(sub_document)
            sub_idx += 1
            end += stride
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))

    def _normalize(self, text):
        raise NotImplementedError
