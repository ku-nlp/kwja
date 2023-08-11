import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rhoknp import Document
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset
from kwja.datamodule.examples import Seq2SeqExample
from kwja.utils.constants import IGNORE_INDEX
from kwja.utils.logging_util import track
from kwja.utils.seq2seq_format import Seq2SeqFormatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Seq2SeqModuleFeatures:
    example_ids: int
    src_text: str
    input_ids: List[int]
    attention_mask: List[int]
    seq2seq_labels: List[int]
    decoder_attention_mask: List[int]


class Seq2SeqDataset(BaseDataset[Seq2SeqExample, Seq2SeqModuleFeatures]):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_src_length: int,
        max_tgt_length: int,
        ext: str = "knp",
        partial_anno_path: Optional[str] = None,
    ) -> None:
        super().__init__(tokenizer, max_src_length)
        self.path = Path(path)

        self.max_src_length: int = max_src_length
        self.max_tgt_length: int = max_tgt_length

        if partial_anno_path is not None:
            with Path(partial_anno_path).open() as f:
                self.partial_anno: Dict[str, Dict[str, Dict[str, str]]] = json.load(f)
        else:
            self.partial_anno = {}

        self.formatter: Seq2SeqFormatter = Seq2SeqFormatter(tokenizer)

        self.documents: List[Document] = self._load_documents(self.path, ext)
        self.examples: List[Seq2SeqExample] = self._load_examples(self.documents)

    @staticmethod
    def _load_documents(document_dir: Path, ext: str = "knp") -> List[Document]:
        documents = []
        for path in track(sorted(document_dir.glob(f"*.{ext}")), description="Loading documents"):
            # TODO: fix document files that raise exception
            try:
                documents.append(Document.from_knp(path.read_text()))
            except AssertionError:
                logger.warning(f"{path} is not a valid knp file.")
        return documents

    def _load_examples(self, documents: List[Document]) -> List[Seq2SeqExample]:
        examples: List[Seq2SeqExample] = []
        example_id: int = 0
        for document in track(documents, description="Loading examples"):
            for sentence in document.sentences:
                src_encoding: BatchEncoding = self.tokenizer(
                    sentence.text,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=False,
                    max_length=self.max_src_length,
                )
                if len(src_encoding.input_ids) > self.max_src_length:
                    logger.warning(f"Length of source sentence is too long: {sentence.text}")
                    continue
                mrph_lines: List[List[str]] = self.formatter.sent_to_mrph_lines(sentence)
                tgt_tokens = self.formatter.tokenize(mrph_lines, self.partial_anno.get(sentence.sid, {}))
                tgt_input_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
                tgt_attention_mask: List[int] = [1] * len(tgt_input_ids) + [0] * (
                    self.max_tgt_length - len(tgt_input_ids)
                )
                tgt_input_ids += [self.tokenizer.pad_token_id] * (self.max_tgt_length - len(tgt_input_ids))
                if len(tgt_input_ids) > self.max_tgt_length:
                    logger.warning(f"Length of target sentence is too long: {sentence.text}")
                    continue
                examples.append(
                    Seq2SeqExample(
                        example_id=example_id,
                        src_text=self.formatter.sent_to_text(sentence),
                        src_encoding=src_encoding,
                        tgt_input_ids=tgt_input_ids,
                        tgt_attention_mask=tgt_attention_mask,
                        sid=sentence.sid,
                    )
                )
                example_id += 1
        if len(examples) == 0:
            logger.error(
                f"No examples to process. Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: Seq2SeqExample) -> Seq2SeqModuleFeatures:
        seq2seq_labels: List[int] = [
            (seq2seq_tag if seq2seq_tag != self.tokenizer.pad_token_id else IGNORE_INDEX)
            for seq2seq_tag in example.tgt_input_ids
        ]
        assert len(seq2seq_labels) == self.max_tgt_length

        return Seq2SeqModuleFeatures(
            example_ids=example.example_id,
            src_text=example.src_text,
            input_ids=example.src_encoding.input_ids,
            attention_mask=example.src_encoding.attention_mask,
            seq2seq_labels=seq2seq_labels,
            decoder_attention_mask=example.tgt_attention_mask,
        )
