from pathlib import Path

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
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.documents = self.load_documents(self.path, ext)
        assert len(self) != 0

        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def load_documents(path: Path, ext: str = "knp") -> list[Document]:
        documents = []
        for file_path in sorted(path.glob(f"**/*.{ext}")):
            with file_path.open("rt", errors="replace") as f:
                documents.append(Document.from_knp(f.read()))
        return documents
