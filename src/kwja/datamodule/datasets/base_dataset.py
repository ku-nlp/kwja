import logging
from dataclasses import dataclass
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
        sentence_tokens = [self._get_tokenized_len(sentence) for sentence in document.sentences]
        if sum(sentence_tokens) <= max_token_length:
            return [document]

        splitter = SequenceSplitter(sentence_tokens, max_token_length, stride)
        sub_documents: List[Document] = []
        sub_idx = 0
        for span in splitter.split_with_overlap():
            assert isinstance(span, SpanCandidate)
            sentences = document.sentences[span.start : span.end]
            sub_document = Document.from_sentences(sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=span.stride)
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.doc_id = sub_doc_id
                sub_sentence.sid = sentence.sid
                sub_sentence.misc_comment = sentence.misc_comment
            sub_document.doc_id = sub_doc_id
            sub_documents.append(sub_document)
            sub_idx += 1
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))


@dataclass(frozen=True)
class SpanCandidate:
    stride: int
    length: int
    start: int
    end: int


class SequenceSplitter:
    """
    This class splits a sequence into sub-sequences where items can be overlapped.
    The split sub-sequences satisfy the following conditions:
    1. The union of the sub-sequences covers the original sequence.
    2. The length of each sub-sequence is less than or equal to `max_length`, unless the above is not violated.
    3. The difference of the end index between two consecutive sub-sequences is equal to `stride`, unless the above are not violated.
    4. The length of each sub-sequence is maximized.

    Args:
        sequence_lengths: A list of item lengths.
        max_length: The maximum length of a sub-sequence span.
        stride: The stride of the sub-sequence span.
    """

    def __init__(self, sequence_lengths: List[int], max_length: int, stride: int) -> None:
        self.sequence_lengths = sequence_lengths
        self.max_length = max_length
        self.stride = stride
        self._cumulative_lengths = [0]
        for length in sequence_lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)

    def split_with_overlap(
        self, return_candidates: bool = False
    ) -> Union[Generator[Tuple[SpanCandidate, List[SpanCandidate]], None, None], Generator[SpanCandidate, None, None]]:
        """
        This function splits a sequence into sub-sequences where items can be overlapped.

        Args:
            return_candidates: If True, this function returns a list of candidates for each sub-sequence span.

        Returns:
            A generator of sub-sequence spans.
        """
        prev_start, prev_end = 0, 0
        while prev_end < len(self.sequence_lengths):
            span, candidates = self.search_sub_sequence_span(prev_start, prev_end)
            prev_start, prev_end = span.start, span.end
            if return_candidates:
                yield span, candidates
            else:
                yield span
        return None

    def search_sub_sequence_span(self, prev_start: int, prev_end: int) -> Tuple[SpanCandidate, List[SpanCandidate]]:
        # if self.stride == -1:
        #     start = prev_end
        #     end = self._search_end(start, initial=prev_end + 1)
        #     span = self._gen_span_candidate(start, end, stride=end - prev_end)
        #     return span, [span]

        candidates: List[SpanCandidate] = []
        # search start index
        for start in range(prev_start, len(self.sequence_lengths)):
            if start > prev_end:
                return self._choose_best_candidate(candidates), candidates
            end = self._search_end(start, initial=prev_end + 1)  # search end index
            span = self._gen_span_candidate(start, end, stride=end - prev_end)
            candidates.append(span)
            if span.length > self.max_length:
                # length condition is not satisfied. Try another start index.
                continue
            if span.stride < self.stride:
                # stride condition is not satisfied. Try another start index.
                continue
            if prev_end > 0:
                # non-first span
                end = prev_end + self.stride
                span = self._gen_span_candidate(start, end, stride=self.stride)
                candidates.append(span)
            return span, candidates
        return self._choose_best_candidate(candidates), candidates

    def _search_end(self, start: int, initial: int) -> int:
        end = initial
        while (
            end + 1 <= len(self.sequence_lengths) and self._get_sub_sequence_length(start, end + 1) <= self.max_length
        ):
            end += 1
        return end

    def _gen_span_candidate(self, start: int, end: int, stride: int) -> SpanCandidate:
        return SpanCandidate(stride, self._get_sub_sequence_length(start, end), start, end)

    def _get_sub_sequence_length(self, start: int, end: int) -> int:
        return self._cumulative_lengths[end] - self._cumulative_lengths[start]

    def _choose_best_candidate(self, candidates: List[SpanCandidate]) -> SpanCandidate:
        return sorted(candidates, key=self._get_priority_score, reverse=True)[0]

    def _get_priority_score(self, span: SpanCandidate) -> int:
        if span.length <= self.max_length:
            weight = self.max_length + 1  # can be any larger number
            score = (span.stride == self.stride) * weight + span.length
        else:
            score = -span.length
        return score
