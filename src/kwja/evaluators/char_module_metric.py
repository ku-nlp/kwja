from collections import defaultdict
from typing import Dict, Iterator, List, Union

from rhoknp import Document, Sentence
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from kwja.datamodule.datasets import CharDataset
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id


class CharModuleMetric(Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("predicted_texts", default=[], dist_reduce_fx="cat")
        self.add_state("gold_texts", default=[], dist_reduce_fx="cat")
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")

    def update(self, predicted_texts: List[str], gold_texts: List[str], example_ids: List[int]) -> None:
        """Update the internal state of the metric.

        Args:
            predicted_texts (List[str]): A list of predicted texts in the KNP format.
            gold_texts (List[str]): A list of gold texts in the KNP format.
            example_ids (List[int]): A list of example IDs.
        """
        self.predicted_texts.append(predicted_texts)
        self.gold_texts.append(gold_texts)
        self.example_ids.append(example_ids)

    def compute(self, dataset: CharDataset) -> Dict[str, Union[str, float]]:
        eid_to_did: Dict[int, str] = {e.example_id: e.doc_id for e in dataset.examples}
        doc_ids: List[str] = [eid_to_did[eid] for eid in self.example_ids]
        _ = self._rebuild_split_documents(self.predicted_texts, doc_ids)
        _ = self._rebuild_split_documents(self.gold_texts, doc_ids)
        raise NotImplementedError

    def evaluate(self, predicted_text: str, gold_text: str) -> Dict[str, float]:
        ret = {}
        ret.update(self.evaluate_word_segmentation(predicted_text, gold_text))
        return ret

    @staticmethod
    def evaluate_word_segmentation(predicted_text: str, gold_text: str) -> Dict[str, float]:
        def convert_sentence_to_labels(document: Document) -> List[str]:
            labels = []
            for morpheme in document.morphemes:
                labels.extend(["B"] + ["I"] * (len(morpheme.text) - 1))
            return labels

        pred = Document.from_jumanpp(predicted_text)
        gold = Document.from_jumanpp(gold_text)
        if pred.text != gold.text:
            raise ValueError("The texts are different.")
        pred_labels = convert_sentence_to_labels(pred)
        gold_labels = convert_sentence_to_labels(gold)
        assert len(pred_labels) == len(gold_labels)
        return {
            "word_segmentation/acc": accuracy_score([gold_labels], [pred_labels]),
            "word_segmentation/f1": f1_score(
                [gold_labels],
                [pred_labels],
                mode="strict",
                scheme=IOB2,
                zero_division="warn",
            ),
        }

    @staticmethod
    def _rebuild_split_documents(texts: List[str], doc_ids: List[str]) -> Iterator[Document]:
        orig_did_to_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        for doc_id, text in zip(doc_ids, texts):
            document = Document.from_jumanpp(text)
            orig_doc_id = to_orig_doc_id(doc_id)
            assert document.doc_id == orig_doc_id
            document.doc_id = doc_id
            orig_did_to_sentences[orig_doc_id] += extract_target_sentences(document)

        for sentences in orig_did_to_sentences.values():
            yield Document.from_sentences(sentences)
