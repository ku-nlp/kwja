import sys
from pathlib import Path

import hydra
from rhoknp import Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


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

        self.documents = self.load_documents(self.path, ext)
        assert len(self) != 0
        self.document_ids: list[str] = [doc.doc_id for doc in self.documents]

        if tokenizer_kwargs:
            tokenizer_kwargs = hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial")
        else:
            tokenizer_kwargs = {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def load_documents(path: Path, ext: str = "knp") -> list[Document]:
        documents = []
        for file_path in sorted(path.glob(f"*.{ext}")):
            # TODO: fix document file
            try:
                documents.append(Document.from_knp(file_path.read_text()))
            except AssertionError:
                print(f"{file_path} is not a valid knp file.", file=sys.stderr)
        return documents
