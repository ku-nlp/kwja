from typing import Dict, List

from torchmetrics import Metric


class TypoModuleMetric(Metric):
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

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError
