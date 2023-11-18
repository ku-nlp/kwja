from datetime import datetime
from typing import List, Optional, Sequence

from rhoknp import Document


def create_documents_from_raw_texts(texts: Sequence[str]) -> List[Document]:
    documents: List[Document] = []
    for text in texts:
        raw_text = ""
        doc_id = ""
        for line in text.split("\n"):
            if line.startswith("# D-ID:"):
                doc_id = line.split()[1].split(":")[1]
            else:
                raw_text += line + "\n"
        document = Document.from_raw_text(raw_text)
        document.doc_id = doc_id
        documents.append(document)
    return documents


def add_doc_ids(documents: Sequence[Document], doc_id_prefix: Optional[str] = None) -> None:
    if doc_id_prefix is None:
        doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
    doc_id_width = len(str(len(documents)))
    for idx, document in enumerate(documents):
        if document.doc_id == "":
            document.doc_id = f"{doc_id_prefix}-{idx:0{doc_id_width}}"


def add_sent_ids(documents: Sequence[Document]) -> None:
    sent_id_width = max((len(str(len(doc.sentences))) for doc in documents if not doc.is_senter_required()), default=0)
    for document in documents:
        if document.is_senter_required():
            continue
        for idx, sentence in enumerate(document.sentences):
            if sentence.sent_id == "":
                sentence.sent_id = f"{document.doc_id}-{idx:0{sent_id_width}}"
