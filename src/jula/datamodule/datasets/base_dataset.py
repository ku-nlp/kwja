import logging
from functools import cached_property
from pathlib import Path

from rhoknp import Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.sub_document import to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        document_split_stride: int,
        model_name_or_path: str,
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        ext: str = "knp",
        **kwargs,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length

        self.orig_documents: list[Document] = self._load_documents(self.path, ext)
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
        assert len(self.documents) != 0

    def __len__(self) -> int:
        return len(self.documents)

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

    @cached_property
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> list[Document]:
        cum_lens = [0]
        for sentence in document.sentences:
            num_tokens = len(self.tokenizer.tokenize(" ".join(morpheme.surf for morpheme in sentence.morphemes)))
            cum_lens.append(cum_lens[-1] + num_tokens)
        if cum_lens[-1] <= max_token_length:
            return [document]

        end = 1
        # end を探索
        while end < len(document.sentences) and cum_lens[end + 1] - cum_lens[0] <= max_token_length:
            end += 1

        sub_documents: list[Document] = []
        sub_idx = 0
        while end < len(document.sentences) + 1:
            start = 0
            # start を探索
            while cum_lens[end] - cum_lens[start] > max_token_length:
                start += 1
                if start == end - 1:
                    break

            sub_document = Document.from_sentences(document.sentences[start:end])
            sub_document.doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=stride)
            sub_documents.append(sub_document)
            sub_idx += 1
            end += stride
        return sub_documents
