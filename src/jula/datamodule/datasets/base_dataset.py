from pathlib import Path

from kyoto_reader import Document, KyotoReader

# from rhoknp import Document
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

        # self.documents = self.load_documents(self.path, ext)
        # assert len(self) != 0

        # use kyoto-reader temporarily
        self.reader: KyotoReader = KyotoReader(
            source=path, target_cases=None, target_corefs=None, extract_nes=False
        )
        self.documents: list[Document] = self.reader.process_all_documents()

        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.documents)

    # TODO: fix bugs in from_knp()
    # @staticmethod
    # def load_documents(path: Path, ext: str = "knp") -> list[Document]:
    #     documents = []
    #     for file_path in sorted(path.glob(f"**/*.{ext}")):
    #         with file_path.open("rt") as f:
    #             documents.append(Document.from_knp(f.read()))
    #     return documents
