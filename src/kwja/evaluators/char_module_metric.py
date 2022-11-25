from typing import Dict, List, Union

from rhoknp import Document
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric


class CharModuleMetric(Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("predicted_texts", default=[], dist_reduce_fx="cat")
        self.add_state("gold_texts", default=[], dist_reduce_fx="cat")

    def update(self, predicted_texts: List[str], gold_texts: List[str]) -> None:
        """Update the internal state of the metric.

        Args:
            predicted_texts (List[str]): A list of predicted texts in the KNP format.
            gold_texts (List[str]): A list of gold texts in the KNP format.
        """
        self.predicted_texts.append(predicted_texts)
        self.gold_texts.append(gold_texts)

    def compute(self) -> Dict[str, Union[str, float]]:
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
