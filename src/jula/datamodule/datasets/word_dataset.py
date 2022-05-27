from pathlib import Path

from rhoknp import Document
from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(self, path: str, ext: str = "knp") -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.documents = self.load_documents(self.path, ext)
        assert len(self) != 0

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def load_documents(path: Path, ext: str = "knp") -> list[Document]:
        documents = []
        for file_path in path.glob(f"**/*.{ext}"):
            with file_path.open("rt") as f:
                documents.append(Document.from_knp(f.read()))
        return documents
