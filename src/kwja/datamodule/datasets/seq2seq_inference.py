import logging
from pathlib import Path
from typing import List, Optional

from rhoknp import KNP, Document, Jumanpp
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.seq2seq import Seq2SeqModuleFeatures
from kwja.datamodule.examples import Seq2SeqInferenceExample
from kwja.utils.progress_bar import track
from kwja.utils.reader import chunk_by_document_for_line_by_line_text

logger = logging.getLogger(__name__)

jumanpp = Jumanpp()
knp = KNP()


class Seq2SeqInferenceDataset(Dataset[Seq2SeqModuleFeatures]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_src_length: int,
        max_tgt_length: int,
        senter_file: Optional[Path] = None,
        **_,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_src_length: int = max_src_length
        self.max_tgt_length: int = max_tgt_length

        if senter_file is not None:
            with senter_file.open() as f:
                documents = [
                    Document.from_line_by_line_text(c)
                    for c in track(chunk_by_document_for_line_by_line_text(f), description="Loading documents")
                ]
        else:
            documents = []
        self.examples: List[Seq2SeqInferenceExample] = self._load_example(documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Seq2SeqModuleFeatures:
        return self.encode(self.examples[index])

    def _load_example(self, documents: List[Document]) -> List[Seq2SeqInferenceExample]:
        examples = []
        example_id: int = 0
        for document in track(documents, description="Loading examples"):
            for sentence in document.sentences:
                src_encoding: BatchEncoding = self.tokenizer(
                    "解析：" + sentence.text.strip(),
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=True,
                    max_length=self.max_src_length,
                )
                examples.append(
                    Seq2SeqInferenceExample(
                        example_id=example_id,
                        src_text=sentence.text.strip(),
                        src_encoding=src_encoding,
                        sid=sentence.sid,
                    )
                )
                example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    @staticmethod
    def encode(example: Seq2SeqInferenceExample) -> Seq2SeqModuleFeatures:
        return Seq2SeqModuleFeatures(
            example_ids=example.example_id,
            src_text=example.src_text,
            input_ids=example.src_encoding.input_ids,
            attention_mask=example.src_encoding.attention_mask,
            seq2seq_labels=[],
        )
