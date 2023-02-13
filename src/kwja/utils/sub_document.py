import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union

from rhoknp import Document, Sentence

SUB_DOC_PAT: re.Pattern = re.compile(r"^(?P<did>[a-zA-Z\d\-_]+?)-s(?P<stride>\d+?)i(?P<idx>\d+?)$")


def to_sub_doc_id(doc_id: str, idx: int, stride: int = 1) -> str:
    return f"{doc_id}-s{stride}i{idx}"


def to_orig_doc_id(doc_id: str) -> str:
    match = SUB_DOC_PAT.match(doc_id)
    if match is None:
        return doc_id
    else:
        return match["did"]


def extract_target_sentences(document: Document) -> List[Sentence]:
    return [sentence for sentence in document.sentences if is_target_sentence(sentence)]


def is_target_sentence(sentence: Sentence) -> bool:
    if sentence.document.doc_id is None:
        return True
    match = SUB_DOC_PAT.match(sentence.document.doc_id)
    if match is None:
        return True
    stride = int(match["stride"])
    return sentence in sentence.document.sentences[-stride:]


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
        stride: The stride of the sub-sequence span. -1 to dynamically determine the maximum stride by eliminating overlap.
    """

    def __init__(self, sequence_lengths: List[int], max_length: int, stride: int) -> None:
        self.sequence_lengths = sequence_lengths
        self.max_length = max_length
        self.stride = stride
        self._cumulative_lengths = [0]
        for length in sequence_lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)

    def split_into_spans(
        self, return_candidates: bool = False
    ) -> Iterator[Union[Tuple[SpanCandidate, List[SpanCandidate]], SpanCandidate]]:
        """
        This function splits a sequence into sub-sequences. Each sub-sequence is represented as a span.

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

    def search_sub_sequence_span(self, prev_start: int, prev_end: int) -> Tuple[SpanCandidate, List[SpanCandidate]]:
        if self.stride == -1:
            start = prev_end
            end = self._search_end(start, initial=prev_end + 1)
            span = self._gen_span_candidate(start, end, stride=end - prev_end)
            return span, [span]

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
