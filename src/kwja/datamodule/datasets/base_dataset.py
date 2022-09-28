import logging
from functools import cached_property
from pathlib import Path
from typing import Union

from rhoknp import Document, Sentence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.sub_document import to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        source: Union[Path, str, list[Document]],
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
        self.orig_documents: list[Document]
        if isinstance(source, (Path, str)):
            source = Path(source)
            assert source.is_dir()
            self.orig_documents = self._load_documents(source, ext)
        else:
            self.orig_documents = source
        self.doc_id2document: dict[str, Document] = {}
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
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())

    @staticmethod
    def _load_documents(document_dir: Path, ext: str = "knp") -> list[Document]:
        documents = []
        for path in sorted(document_dir.glob(f"*.{ext}")):
            # TODO: fix document files that raise exception
            try:
                documents.append(Document.from_knp(path.read_text()))
            except AssertionError:
                logger.warning(f"{path} is not a valid knp file.")
        return documents

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> list[Document]:
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

        sub_documents: list[Document] = []
        sub_idx = 0
        while end <= len(document.sentences):
            start = 0
            # start を探索
            while cum_lens[end] - cum_lens[start] > max_token_length:
                if start == end - 1:
                    break
                start += 1

            # TODO: fix rhoknp to keep comments in sentence
            comments = [s.comment for s in document.sentences[start:end]]
            sub_document = Document.from_sentences(document.sentences[start:end])
            for comment, sentence in zip(comments, sub_document.sentences):
                sentence.comment = comment
            sub_document.doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=stride)
            sub_documents.append(sub_document)
            sub_idx += 1
            end += stride
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(
            self.tokenizer([m.text for m in source.morphemes], add_special_tokens=False, is_split_into_words=True)[
                "input_ids"
            ]
        )
