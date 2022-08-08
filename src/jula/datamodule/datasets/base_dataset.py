import logging
from functools import cached_property
from pathlib import Path

from rhoknp import Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name_or_path: str,
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        ext: str = "knp",
        **kwargs,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.doc_id2document = self.load_documents(self.path, ext)
        assert len(self.documents) != 0

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def load_documents(path: Path, ext: str = "knp") -> dict[str, Document]:
        doc_id2document: dict[str, Document] = {}
        for file_path in sorted(path.glob(f"*.{ext}")):
            # TODO: fix document file
            try:
                document = Document.from_knp(file_path.read_text())
                doc_id2document[document.doc_id] = document
            except AssertionError:
                logger.error(f"{file_path} is not a valid knp file.")
        return doc_id2document

    @cached_property
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())
