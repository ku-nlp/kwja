import logging
from typing import List

from omegaconf import ListConfig
from torch.utils.data import Dataset
from rhoknp import KNP, Jumanpp
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.examples import Seq2SeqInferenceExample
from kwja.datamodule.datasets.seq2seq import Seq2SeqModuleFeatures
from kwja.utils.progress_bar import track

logger = logging.getLogger(__name__)

jumanpp = Jumanpp()
knp = KNP()


class Seq2SeqInferenceDataset(Dataset[Seq2SeqModuleFeatures]):
    def __init__(
        self,
        texts: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_src_length: int,
        max_tgt_length: int,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_src_length: int = max_src_length
        self.max_tgt_length: int = max_tgt_length

        self.examples: List[Seq2SeqInferenceExample] = []

        example_id: int = 0
        for text in track(texts, description="Loading documents"):
            text = text.strip()
            src_encoding: BatchEncoding = self.tokenizer(
                "解析：" + text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
                max_length=self.max_src_length,
            )
            self.examples.append(
                Seq2SeqInferenceExample(
                    example_id=example_id,
                    src_text=text,
                    src_encoding=src_encoding,
                )
            )
            example_id += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Seq2SeqModuleFeatures:
        return self.encode(self.examples[index])

    @staticmethod
    def encode(example: Seq2SeqInferenceExample) -> Seq2SeqModuleFeatures:
        return Seq2SeqModuleFeatures(
            example_ids=example.example_id,
            src_text=example.src_text,
            input_ids=example.src_encoding.input_ids,
            attention_mask=example.src_encoding.attention_mask,
            seq2seq_labels=[]
        )
