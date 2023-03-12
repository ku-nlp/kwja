import logging
from datetime import datetime
from typing import List, Optional

from omegaconf import ListConfig
from rhoknp import KNP, Document, Jumanpp, RegexSenter
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import kwja
from kwja.datamodule.datasets.seq2seq import Seq2SeqModuleFeatures
from kwja.datamodule.examples import Seq2SeqInferenceExample
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
        doc_id_prefix: Optional[str] = None,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_src_length: int = max_src_length
        self.max_tgt_length: int = max_tgt_length

        documents: List[Document] = self._build_documents_from_texts(list(texts), doc_id_prefix)
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

    @staticmethod
    def _build_documents_from_texts(texts: List[str], doc_id_prefix: Optional[str]) -> List[Document]:
        senter = RegexSenter()
        # split text into sentences
        documents: List[Document] = [
            senter.apply_to_document(text) for text in track(texts, description="Loading documents")
        ]
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id_width = len(str(len(documents)))
        sent_id_width = max((len(str(len(doc.sentences))) for doc in documents), default=0)
        for document_index, document in enumerate(documents):
            document.doc_id = f"{doc_id_prefix}-{document_index:0{doc_id_width}}"
            for sentence_index, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sentence_index:0{sent_id_width}}"
                sentence.misc_comment = f"kwja:{kwja.__version__}"
        return documents
