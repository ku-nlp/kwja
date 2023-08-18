from typing import Iterator, List, TextIO

from rhoknp import Sentence
from rhoknp.utils.comment import is_comment_line


def chunk_by_document_for_line_by_line_text(f: TextIO) -> Iterator[str]:
    sentences: List[Sentence] = []
    prev_doc_id = ""
    for sentence_text in chunk_by_sentence_for_line_by_line_text(f):
        sentence = Sentence.from_raw_text(sentence_text)
        if sentence.doc_id != prev_doc_id:
            if sentences:
                yield "".join(sentence.to_raw_text() for sentence in sentences)
                sentences = []
        sentences.append(sentence)
        prev_doc_id = sentence.doc_id
    yield "".join(sentence.to_raw_text() for sentence in sentences)


def chunk_by_sentence_for_line_by_line_text(f: TextIO) -> Iterator[str]:
    buffer = []
    for line in f:
        if line.strip() == "":
            continue
        buffer.append(line)
        if is_comment_line(line) is False:
            yield "".join(buffer)
            buffer = []
    if buffer:
        yield "".join(buffer)
