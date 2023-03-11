from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from rhoknp import Document, Sentence
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

from kwja.callbacks.utils import convert_senter_predictions_into_tags, set_sentences
from kwja.datamodule.datasets import SenterDataset
from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique
from kwja.utils.sub_document import to_orig_doc_id


class SenterModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "sent_segmentation_predictions",
    )

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Optional[SenterDataset] = None
        self.example_ids: torch.Tensor
        self.sent_segmentation_predictions: torch.Tensor

    def compute(self) -> Dict[str, float]:
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            setattr(self, state_name, state[sorted_indices])

        predicted_documents, gold_documents = self._build_documents()

        return self.compute_sent_segmentation_metrics(predicted_documents, gold_documents)

    def _build_documents(self) -> Tuple[List[Document], List[Document]]:
        assert self.dataset is not None, "dataset isn't set"

        doc_id2predicted_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        doc_id2gold_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        special_ids = set(self.dataset.tokenizer.all_special_ids) - {self.dataset.tokenizer.unk_token_id}
        for example_id, sent_segmentation_predictions in zip(
            self.example_ids.tolist(),
            self.sent_segmentation_predictions.tolist(),
        ):
            example = self.dataset.examples[example_id]
            gold_document = self.dataset.doc_id2document[example.doc_id]
            predicted_document = Document(gold_document.text)
            predicted_document.doc_id = gold_document.doc_id

            assert len(example.encoding.input_ids) == len(sent_segmentation_predictions)
            sent_segmentation_tags = convert_senter_predictions_into_tags(
                sent_segmentation_predictions, example.encoding.input_ids, special_ids
            )
            set_sentences(predicted_document, sent_segmentation_tags)

            orig_doc_id = to_orig_doc_id(gold_document.doc_id)
            # Because the stride is always -1, every sentence is a target sentence.
            doc_id2predicted_sentences[orig_doc_id].extend(predicted_document.sentences)
            doc_id2gold_sentences[orig_doc_id].extend(gold_document.sentences)
        predicted_documents = self._convert_doc_id2sentences_into_documents(doc_id2predicted_sentences)
        gold_documents = self._convert_doc_id2sentences_into_documents(doc_id2gold_sentences)
        return predicted_documents, gold_documents

    @staticmethod
    def _convert_doc_id2sentences_into_documents(doc_id2sentences: Dict[str, List[Sentence]]) -> List[Document]:
        return [Document.from_sentences(ss) for ss in doc_id2sentences.values()]

    def compute_sent_segmentation_metrics(
        self, predicted_documents: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        labels = [self._convert_document_into_segmentation_tags(d) for d in gold_documents]
        predictions = [self._convert_document_into_segmentation_tags(d) for d in predicted_documents]
        return {
            "sent_segmentation_accuracy": accuracy_score(y_true=labels, y_pred=predictions),
            "sent_segmentation_f1": f1_score(
                y_true=labels,
                y_pred=predictions,
                mode="strict",
                scheme=IOB2,
            ),
        }

    @staticmethod
    def _convert_document_into_segmentation_tags(document: Document) -> List[str]:
        segmentation_tags = []
        for sentence in document.sentences:
            segmentation_tags.extend(["B"] + ["I"] * (len(sentence.text) - 1))
        return segmentation_tags
