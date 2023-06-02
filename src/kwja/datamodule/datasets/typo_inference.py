from collections import defaultdict
from typing import Dict, List

from omegaconf import ListConfig
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset
from kwja.datamodule.datasets.typo import TypoModuleFeatures
from kwja.datamodule.examples import TypoInferenceExample
from kwja.utils.constants import DUMMY_TOKEN
from kwja.utils.logging_util import track


class TypoInferenceDataset(BaseDataset[TypoInferenceExample, TypoModuleFeatures]):
    def __init__(
        self,
        texts: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)

        self.examples: List[TypoInferenceExample] = []
        self.stash: Dict[int, List[str]] = defaultdict(list)
        example_id = 0
        for text in track(texts, description="Loading documents"):
            text = text.strip()
            if len(self.tokenizer.tokenize(text)) == len(text) <= max_seq_length - 3:  # -3: [CLS], DUMMY_TOKEN, [SEP]
                self.examples.append(TypoInferenceExample(example_id=example_id, pre_text=text))
                example_id += 1
            else:
                self.stash[example_id].append(text)
        if len(self.examples) == 0:
            # len(self.examples) == 0だとwriterが呼ばれないのでダミーを追加
            self.examples.append(TypoInferenceExample(example_id=example_id, pre_text=""))

    def encode(self, example: TypoInferenceExample) -> TypoModuleFeatures:
        encoding: BatchEncoding = self.tokenizer(
            example.pre_text + DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        return TypoModuleFeatures(
            example_ids=example.example_id,
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            kdr_labels=[],
            ins_labels=[],
        )
