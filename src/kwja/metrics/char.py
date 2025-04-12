from collections import defaultdict
from typing import Optional

import torch
from rhoknp import Document, Sentence
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

from kwja.callbacks.utils import convert_char_predictions_into_tags, set_morphemes, set_sentences
from kwja.datamodule.datasets import CharDataset
from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique
from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_OP_TAGS
from kwja.utils.sub_document import to_orig_doc_id


class CharModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "sent_segmentation_predictions",
        "word_segmentation_predictions",
        "word_norm_op_predictions",
        "word_norm_op_labels",
    )

    def __init__(self, max_seq_length: int) -> None:
        super().__init__(max_seq_length)
        self.dataset: Optional[CharDataset] = None
        self.example_ids: torch.Tensor
        self.sent_segmentation_predictions: torch.Tensor
        self.word_segmentation_predictions: torch.Tensor
        self.word_norm_op_predictions: torch.Tensor
        self.word_norm_op_labels: torch.Tensor

    def _pad(self, kwargs: dict[str, torch.Tensor]) -> None:
        for key, value in kwargs.items():
            if key in {"example_ids"} or value.numel() == 0:
                continue
            else:
                dims = [1]
            for dim in dims:
                size = [self.max_seq_length - s if i == dim else s for i, s in enumerate(value.size())]
                padding = torch.zeros(size, dtype=value.dtype, device=value.device)
                value = torch.cat([value, padding], dim=dim)
            kwargs[key] = value

    def compute(self) -> dict[str, float]:
        if isinstance(self.example_ids, torch.Tensor) is False:
            self.example_ids = torch.cat(self.example_ids, dim=0)  # type: ignore
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if isinstance(state, torch.Tensor) is False:
                state = torch.cat(state, dim=0)
            setattr(self, state_name, state[sorted_indices])

        predicted_documents, partly_gold_documents, gold_documents = self._build_documents()

        return {
            **self.compute_sent_segmentation_metrics(predicted_documents, gold_documents),
            **self.compute_word_segmentation_metrics(partly_gold_documents, gold_documents),
            **self.compute_word_normalization_metrics(self.word_norm_op_predictions, self.word_norm_op_labels),
        }

    def _build_documents(self) -> tuple[list[Document], list[Document], list[Document]]:
        assert self.dataset is not None, "dataset isn't set"

        doc_id2predicted_sentences: dict[str, list[Sentence]] = defaultdict(list)
        doc_id2partly_gold_sentences: dict[str, list[Sentence]] = defaultdict(list)
        doc_id2gold_sentences: dict[str, list[Sentence]] = defaultdict(list)
        special_ids = {
            getattr(self.dataset.tokenizer, f"{prefix}_token_id")
            for prefix in ["bos", "eos", "sep", "pad", "cls", "mask"]
        }
        for (
            example_id,
            sent_segmentation_predictions,
            word_segmentation_predictions,
            word_norm_op_predictions,
            _,  # word_norm_op_labels
        ) in zip(*[getattr(self, state_name).tolist() for state_name in self.STATE_NAMES]):
            example = self.dataset.examples[example_id]
            gold_document = self.dataset.doc_id2document[example.doc_id]
            predicted_document = Document(gold_document.text)
            predicted_document.doc_id = gold_document.doc_id
            partly_gold_document = Document.from_jumanpp(gold_document.to_jumanpp())
            partly_gold_document.doc_id = gold_document.doc_id

            assert (
                len(example.encoding.input_ids)
                == len(sent_segmentation_predictions)
                == len(word_segmentation_predictions)
                == len(word_norm_op_predictions)
            )
            sent_segmentation_tags, word_segmentation_tags, word_norm_op_tags = convert_char_predictions_into_tags(
                sent_segmentation_predictions,
                word_segmentation_predictions,
                word_norm_op_predictions,
                [i for i, input_id in enumerate(example.encoding.input_ids) if input_id not in special_ids],
            )
            set_sentences(predicted_document, sent_segmentation_tags)
            set_morphemes(partly_gold_document, word_segmentation_tags, word_norm_op_tags)

            orig_doc_id = to_orig_doc_id(gold_document.doc_id)
            # Every sentence is a target sentence because document_split_stride is always -1
            doc_id2predicted_sentences[orig_doc_id].extend(predicted_document.sentences)
            doc_id2partly_gold_sentences[orig_doc_id].extend(partly_gold_document.sentences)
            doc_id2gold_sentences[orig_doc_id].extend(gold_document.sentences)
        predicted_documents = self._convert_doc_id2sentences_into_documents(
            doc_id2predicted_sentences, from_sentences=True
        )
        partly_gold_documents = self._convert_doc_id2sentences_into_documents(doc_id2partly_gold_sentences)
        gold_documents = self._convert_doc_id2sentences_into_documents(doc_id2gold_sentences)
        return predicted_documents, partly_gold_documents, gold_documents

    @staticmethod
    def _convert_doc_id2sentences_into_documents(
        doc_id2sentences: dict[str, list[Sentence]],
        from_sentences: bool = False,
    ) -> list[Document]:
        # Build documents that do not have clauses, phrases, or base phrases, but morphemes only
        return [
            (
                Document.from_jumanpp("".join(s.to_jumanpp() for s in ss))
                if from_sentences is False
                else Document.from_sentences(ss)
            )
            for ss in doc_id2sentences.values()
        ]

    def compute_sent_segmentation_metrics(
        self, predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        labels = [self._convert_document_into_sent_segmentation_tags(d) for d in gold_documents]
        predictions = [self._convert_document_into_sent_segmentation_tags(d) for d in predicted_documents]
        return {
            "sent_segmentation_accuracy": accuracy_score(y_true=labels, y_pred=predictions),
            "sent_segmentation_f1": f1_score(y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2).item(),
        }

    @staticmethod
    def _convert_document_into_sent_segmentation_tags(document: Document) -> list[str]:
        sent_segmentation_tags = []
        for sentence in document.sentences:
            sent_segmentation_tags.extend(["B"] + ["I"] * (len(sentence.text) - 1))
        return sent_segmentation_tags

    def compute_word_segmentation_metrics(
        self, predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        labels = [self._convert_document_into_word_segmentation_tags(d) for d in gold_documents]
        predictions = [self._convert_document_into_word_segmentation_tags(d) for d in predicted_documents]
        return {
            "word_segmentation_accuracy": accuracy_score(y_true=labels, y_pred=predictions),
            "word_segmentation_f1": f1_score(y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2).item(),
        }

    @staticmethod
    def _convert_document_into_word_segmentation_tags(document: Document) -> list[str]:
        word_segmentation_tags = []
        for morpheme in document.morphemes:
            word_segmentation_tags.extend(["B"] + ["I"] * (len(morpheme.text) - 1))
        return word_segmentation_tags

    @staticmethod
    def compute_word_normalization_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        ignored_indices = labels.eq(IGNORE_INDEX)
        predictions = predictions[~ignored_indices]
        labels = labels[~ignored_indices]

        metrics: dict[str, float] = {
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
