import re

from rhoknp import Document, Sentence

SUB_DOC_PAT: re.Pattern[str] = re.compile(r"^(?P<did>[a-zA-Z\d\-_]+?)-split(?P<stride>[1-9])(?P<idx>\d{2})$")


def is_target_sentence(sentence: Sentence) -> bool:
    document = sentence.document
    match = SUB_DOC_PAT.match(document.doc_id)
    if match is None:
        return True
    stride = int(match.group("stride"))
    idx = int(match.group("idx"))
    if idx == 0:
        return True
    else:
        return sentence in document.sentences[-stride:]


def extract_target_sentences(document: Document) -> list[Sentence]:
    return [sentence for sentence in document.sentences if is_target_sentence(sentence)]


def to_orig_doc_id(doc_id: str) -> str:
    match = SUB_DOC_PAT.match(doc_id)
    if match is None:
        return doc_id
    else:
        return match.group("did")


def to_sub_doc_id(doc_id: str, idx: int, stride: int = 1) -> str:
    return f"{doc_id}-split{stride}{idx:02}"
