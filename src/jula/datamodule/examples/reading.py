from rhoknp import Document

from jula.utils.reading_aligner import ReadingAligner


class ReadingExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.readings: list[str] = []

    def load(self, document: Document, aligner: ReadingAligner) -> None:
        self.doc_id = document.doc_id
        for _, reading in aligner.align(document):
            self.readings.append(reading)
