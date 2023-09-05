import logging
from pathlib import Path
from typing import List, Optional

from rhoknp import Document
from rhoknp.utils.reader import chunk_by_document
from transformers import PreTrainedTokenizerFast

from kwja.datamodule.datasets.base import BaseDataset
from kwja.datamodule.datasets.seq2seq import Seq2SeqModuleFeatures
from kwja.datamodule.examples import Seq2SeqInferenceExample
from kwja.utils.logging_util import track
from kwja.utils.seq2seq_format import Seq2SeqFormatter

logger = logging.getLogger(__name__)


class Seq2SeqInferenceDataset(BaseDataset[Seq2SeqInferenceExample, Seq2SeqModuleFeatures]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_src_length: int,
        max_tgt_length: int,
        juman_file: Optional[Path] = None,
        **_,
    ) -> None:
        super().__init__(tokenizer, max_src_length)
        self.max_src_length: int = max_src_length
        self.max_tgt_length: int = max_tgt_length

        self.formatter: Seq2SeqFormatter = Seq2SeqFormatter(tokenizer)

        if juman_file is not None:
            with juman_file.open() as f:
                documents = [
                    Document.from_jumanpp(c) for c in track(chunk_by_document(f), description="Loading documents")
                ]
        else:
            documents = []
        self.examples: List[Seq2SeqInferenceExample] = self._load_example(documents)

    def _load_example(self, documents: List[Document]) -> List[Seq2SeqInferenceExample]:
        examples = []
        example_id: int = 0
        for document in track(documents, description="Loading examples"):
            for sentence in document.sentences:
                src_tokens: List[str] = self.formatter.get_src_tokens(sentence)
                src_input_ids: List[int] = self.tokenizer.convert_tokens_to_ids(src_tokens) + [
                    self.tokenizer.eos_token_id
                ]
                src_attention_mask: List[int] = [1] * len(src_input_ids)
                src_input_ids += [self.tokenizer.pad_token_id] * (self.max_src_length - len(src_input_ids))
                src_attention_mask += [0] * (self.max_src_length - len(src_attention_mask))
                examples.append(
                    Seq2SeqInferenceExample(
                        example_id=example_id,
                        surfs=self.formatter.get_surfs(sentence),
                        src_input_ids=src_input_ids,
                        src_attention_mask=src_attention_mask,
                        sid=sentence.sid,
                    )
                )
                example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: Seq2SeqInferenceExample) -> Seq2SeqModuleFeatures:
        return Seq2SeqModuleFeatures(
            example_ids=example.example_id,
            surfs=example.surfs,
            input_ids=example.src_input_ids,
            attention_mask=example.src_attention_mask,
            seq2seq_labels=[],
        )
