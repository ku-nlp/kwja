from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from rhoknp import Document, Sentence
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

from kwja.callbacks.utils import convert_predictions_into_tags, set_morphemes
from kwja.datamodule.datasets import CharDataset
from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique
from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_OP_TAGS
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id


class CharModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "word_segmentation_predictions",
        "word_norm_op_predictions",
        "word_norm_op_labels",
    )

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Optional[CharDataset] = None
        self.example_ids: torch.Tensor
        self.word_segmentation_predictions: torch.Tensor
        self.word_norm_op_predictions: torch.Tensor
        self.word_norm_op_labels: torch.Tensor

    def compute(self) -> Dict[str, float]:
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            setattr(self, state_name, state[sorted_indices])

        predicted_documents, gold_documents = self._build_documents()

        return {
            **self.compute_word_segmentation_metrics(predicted_documents, gold_documents),
            **self.compute_word_normalization_metrics(self.word_norm_op_predictions, self.word_norm_op_labels),
        }

    def _build_documents(self) -> Tuple[List[Document], List[Document]]:
        assert self.dataset is not None, "dataset isn't set"

        doc_id2predicted_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        doc_id2gold_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        special_ids = set(self.dataset.tokenizer.all_special_ids) - {self.dataset.tokenizer.unk_token_id}
        for example_id, word_segmentation_predictions, word_norm_op_predictions in zip(
            self.example_ids.tolist(),
            self.word_segmentation_predictions.tolist(),
            self.word_norm_op_predictions.tolist(),
        ):
            example = self.dataset.examples[example_id]
            gold_document = self.dataset.doc_id2document[example.doc_id]
            predicted_document = Document.from_jumanpp(gold_document.to_jumanpp())
            predicted_document.doc_id = gold_document.doc_id

            assert (
                len(example.encoding.input_ids) == len(word_segmentation_predictions) == len(word_norm_op_predictions)
            )
            word_segmentation_tags, word_norm_op_tags = convert_predictions_into_tags(
                word_segmentation_predictions, word_norm_op_predictions, example.encoding.input_ids, special_ids
            )
            set_morphemes(predicted_document, word_segmentation_tags, word_norm_op_tags)

            orig_doc_id = to_orig_doc_id(gold_document.doc_id)
            for sentence in extract_target_sentences(predicted_document):
                doc_id2predicted_sentences[orig_doc_id].append(sentence)
            for sentence in extract_target_sentences(gold_document):
                doc_id2gold_sentences[orig_doc_id].append(sentence)
        predicted_documents = self._convert_doc_id2sentences_into_documents(doc_id2predicted_sentences)
        gold_documents = self._convert_doc_id2sentences_into_documents(doc_id2gold_sentences)
        return predicted_documents, gold_documents

    @staticmethod
    def _convert_doc_id2sentences_into_documents(doc_id2sentences: Dict[str, List[Sentence]]) -> List[Document]:
        # Build documents that do not have clauses, phrases, or base phrases, but morphemes only
        return [Document.from_jumanpp("".join(s.to_jumanpp() for s in ss)) for ss in doc_id2sentences.values()]

    def compute_word_segmentation_metrics(
        self, predicted_documents: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        labels = [self._convert_document_into_segmentation_tags(d) for d in gold_documents]
        predictions = [self._convert_document_into_segmentation_tags(d) for d in predicted_documents]
        return {
            "word_segmentation_accuracy": accuracy_score(y_true=labels, y_pred=predictions),
            "word_segmentation_f1": f1_score(
                y_true=labels,
                y_pred=predictions,
                mode="strict",
                scheme=IOB2,
            ),
        }

    @staticmethod
    def _convert_document_into_segmentation_tags(document: Document) -> List[str]:
        segmentation_tags = []
        for morpheme in document.morphemes:
            segmentation_tags.extend(["B"] + ["I"] * (len(morpheme.text) - 1))
        return segmentation_tags

    @staticmethod
    def compute_word_normalization_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        ignored_indices = labels.eq(IGNORE_INDEX)
        predictions = predictions[~ignored_indices]
        labels = labels[~ignored_indices]

        metrics: Dict[str, float] = {
            "word_normalization_accuracy": accuracy_score(y_true=labels, y_pred=predictions).item()
        }

        keep_indices = predictions.eq(WORD_NORM_OP_TAGS.index("K"))
        if (~keep_indices).sum().item() == 0:
            precision = 0.0
        else:
            precision = accuracy_score(y_true=labels[~keep_indices], y_pred=predictions[~keep_indices]).item()

        keep_indices = labels.eq(WORD_NORM_OP_TAGS.index("K"))
        if (~keep_indices).sum().item() == 0:
            recall = 0.0
        else:
            recall = accuracy_score(y_true=labels[~keep_indices], y_pred=predictions[~keep_indices]).item()

        if (precision + recall) == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        metrics["word_normalization_f1"] = f1

        for word_norm_op_index, word_norm_op_tag in enumerate(WORD_NORM_OP_TAGS):
            indices = predictions.eq(word_norm_op_index)
            if indices.sum().item() == 0:
                precision = 0.0
            else:
                precision = accuracy_score(y_true=labels[indices], y_pred=predictions[indices]).item()

            indices = labels.eq(word_norm_op_index)
            if indices.sum().item() == 0:
                recall = 0.0
            else:
                recall = accuracy_score(y_true=labels[indices], y_pred=predictions[indices]).item()

            if (precision + recall) == 0.0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            metrics[f"word_normalization_f1:{word_norm_op_tag}"] = f1
        return metrics
