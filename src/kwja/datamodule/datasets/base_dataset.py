import logging
from functools import cached_property
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

from rhoknp import Document, Sentence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.progress_bar import track
from kwja.utils.sub_document import to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        source: Union[Path, str, List[Document]],
        document_split_stride: int,
        model_name_or_path: str,
        max_seq_length: int,
        tokenizer_kwargs: dict,
        ext: str = "knp",
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length
        self.orig_documents: List[Document]
        if isinstance(source, (Path, str)):
            source = Path(source)
            assert source.is_dir()
            self.orig_documents = self._load_documents(source, ext)
        else:
            self.orig_documents = source
        self.doc_id2document: Dict[str, Document] = {}
        for orig_document in track(self.orig_documents, description="Splitting documents"):
            orig_document = self._normalize(orig_document)
            self.doc_id2document.update(
                {
                    document.doc_id: document
                    for document in self._split_document(
                        orig_document,
                        self.max_seq_length - len(self.tokenizer.additional_special_tokens) - 2,  # -2: [CLS] and [SEP]
                        stride=document_split_stride,
                    )
                }
            )

    @cached_property
    def documents(self) -> List[Document]:
        return list(self.doc_id2document.values())

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

    def _normalize(self, document: Document) -> Document:
        return document

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> List[Document]:
        cum_lens = [0]
        for sentence in document.sentences:
            num_tokens = self._get_tokenized_len(sentence)
            cum_lens.append(cum_lens[-1] + num_tokens)
        if cum_lens[-1] <= max_token_length:
            return [document]

        end = 1
        # end を探索
        while end < len(document.sentences) and cum_lens[end + 1] - cum_lens[0] <= max_token_length:
            end += 1

        sub_documents: List[Document] = []
        sub_idx = 0
        while end <= len(document.sentences):
            start = 0
            # start を探索
            while cum_lens[end] - cum_lens[start] > max_token_length:
                if start == end - 1:
                    break
                start += 1

            sentences = document.sentences[start:end]
            sub_document = Document.from_sentences(sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=stride)
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.doc_id = sub_doc_id
                sub_sentence.sid = sentence.sid
            sub_document.doc_id = sub_doc_id
            sub_documents.append(sub_document)
            sub_idx += 1
            stride = max(min(stride, len(document.sentences) - end), 1)
            end += stride
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))


def split_with_overlap(
    sequence_lengths: List[int], max_length: int, stride: int
) -> Generator[Tuple[int, int], None, None]:
    """
    This function splits a sequence into sub-sequences where items can be overlapped.
    The split sub-sequences satisfy the following conditions:
    1. The union of the sub-sequences covers the original sequence.
    2. The length of each sub-sequence is less than or equal to `max_length`, unless the above is not violated.
    3. The difference of the end index between two consecutive sub-sequences is equal to `stride`, unless the above are not violated.
    4. The length of each sub-sequence is maximized.

    Args:
        sequence_lengths: A list of item lengths.
        max_length: The maximum length of a sub-sequence span.
        stride: The stride of the sub-sequence span.

    Returns:
        A generator of sub-sequence spans.
    """
    prev_start, prev_end = 0, 0
    while prev_end < len(sequence_lengths):
        start, end = search_sub_document_span(sequence_lengths, max_length, stride, prev_start, prev_end)
        prev_start, prev_end = start, end
        yield start, end
    return None


def search_sub_document_span(
    sequence_lengths: List[int], max_length: int, stride: int, prev_start, prev_end
) -> Tuple[int, int]:
    buff: List[Tuple[int, int]] = []
    # search start index
    for start in range(prev_start, len(sequence_lengths)):
        if start > prev_end:
            return buff[-1][0], buff[-1][1]  # return the last span
        # search end index
        end = prev_end + 1
        buff.append((start, end))
        if sum(sequence_lengths[start:end]) > max_length:
            continue  # even a single item exceeds the max length
        while end + 1 <= len(sequence_lengths) and sum(sequence_lengths[start : end + 1]) <= max_length:
            end += 1
        if end - prev_end >= stride:
            if prev_end == 0:
                return start, end  # first span
            else:
                return start, prev_end + stride
        else:
            buff.append((start, end))  # stride condition is not satisfied
    return buff[-1][0], buff[-1][1]  # return the last span
