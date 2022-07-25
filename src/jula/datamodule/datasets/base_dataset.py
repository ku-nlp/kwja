import sys
from functools import cached_property
from pathlib import Path

import hydra
from rhoknp import Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        ext: str = "knp",
        **kwargs,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.doc_id2document = self.load_documents(self.path, ext)
        assert len(self.documents) != 0

        if tokenizer_kwargs:
            tokenizer_kwargs = hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial")
        else:
            tokenizer_kwargs = {}
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
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
                print(f"{file_path} is not a valid knp file.", file=sys.stderr)
        return doc_id2document

    @cached_property
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())
