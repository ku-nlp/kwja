import logging
from typing import Union

from rhoknp import Document

from kwja.utils.reading import ReadingAligner

logger = logging.getLogger(__name__)


class ReadingExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.readings: Union[list[str], None] = []

    def load(self, document: Document, aligner: ReadingAligner) -> None:
        self.doc_id = document.doc_id
        try:
            self.readings = []
            for _, reading in aligner.align(document):
                self.readings.append(reading)
        except Exception as e:
            logger.warning(e)
            self.readings = None
