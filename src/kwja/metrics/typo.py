from typing import Dict, List, Optional, Tuple

import torch
from Levenshtein import opcodes

from kwja.callbacks.utils import apply_edit_operations, convert_predictions_into_typo_corr_op_tags
from kwja.datamodule.datasets import TypoDataset
from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique


class TypoModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "kdr_predictions",
        "kdr_probabilities",
        "ins_predictions",
        "ins_probabilities",
    )

    def __init__(self, confidence_thresholds: Tuple[float, ...] = (0.0, 0.8, 0.9)) -> None:
        super().__init__()
        self.confidence_thresholds = confidence_thresholds
        self.dataset: Optional[TypoDataset] = None
        self.example_ids: torch.Tensor
        self.kdr_predictions: torch.Tensor
        self.kdr_probabilities: torch.Tensor
        self.ins_predictions: torch.Tensor
        self.ins_probabilities: torch.Tensor

    def compute(self) -> Dict[str, float]:
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            setattr(self, state_name, state[sorted_indices])

        metrics: Dict[str, float] = {}
        for confidence_threshold in self.confidence_thresholds:
            texts = self._build_texts(confidence_threshold)
            metrics.update(self.compute_typo_correction_metrics(texts, confidence_threshold))
        return metrics

    def _build_texts(self, confidence_threshold: float) -> List[Tuple[str, str, str]]:
        example_id2texts = {}
        for example_id, kdr_predictions, kdr_probabilities, ins_predictions, ins_probabilities in zip(
            self.example_ids.tolist(),
            self.kdr_predictions.tolist(),
            self.kdr_probabilities.tolist(),
            self.ins_predictions.tolist(),
            self.ins_probabilities.tolist(),
        ):
            assert self.dataset is not None, "typo dataset isn't set"

            example = self.dataset.examples[example_id]
            seq_len: int = len(example.pre_text)
            if seq_len == 0:
                continue

            args = (confidence_threshold, self.dataset.token2token_id, self.dataset.token_id2token)
            kdr_tags = convert_predictions_into_typo_corr_op_tags(kdr_predictions, kdr_probabilities, "R", *args)
            ins_tags = convert_predictions_into_typo_corr_op_tags(ins_predictions, ins_probabilities, "I", *args)

            # the prediction of the first token (= [CLS]) is excluded.
            # the prediction of the dummy token at the end is used for insertion only.
            predicted_text = apply_edit_operations(
                example.pre_text, kdr_tags[1 : seq_len + 1], ins_tags[1 : seq_len + 2]
            )
            example_id2texts[example_id] = (example.pre_text, predicted_text, example.post_text)
        return list(example_id2texts.values())

    def compute_typo_correction_metrics(
        self, texts: List[Tuple[str, str, str]], confidence_threshold: float
    ) -> Dict[str, float]:
        tp, fp, fn = 0, 0, 0
        for pre_text, predicted_text, gold_text in texts:
            predicted_diffs = self._get_diffs(pre_text, predicted_text)
            gold_diffs = self._get_diffs(pre_text, gold_text)
            intersection = []
            queue = [*gold_diffs]
            for predicted_diff in predicted_diffs:
                if predicted_diff in queue:
                    intersection.append(predicted_text)
                    queue.remove(predicted_diff)
            assert (
                len(predicted_diffs) - len(intersection) >= 0 and len(gold_diffs) - len(intersection) >= 0
            ), "invalid computation of tp"
            tp += len(intersection)
            fp += len(predicted_diffs) - len(intersection)
            fn += len(gold_diffs) - len(intersection)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0.0:
            f1 = 0.0
            f05 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        return {
            f"typo_correction_{confidence_threshold}_precision": precision,
            f"typo_correction_{confidence_threshold}_recall": recall,
            f"typo_correction_{confidence_threshold}_f1": f1,
            f"typo_correction_{confidence_threshold}_f0.5": f05,
        }

    @staticmethod
    def _get_diffs(pre_text: str, post_text: str) -> List[Tuple[str, str]]:
        diffs: List[Tuple[str, str]] = []
        for tag, i1, i2, j1, j2 in opcodes(pre_text, post_text):
            if tag == "delete":
                for pre_char in pre_text[i1:i2]:
                    diffs.append((pre_char, ""))
            elif tag == "insert":
                for post_char in post_text[j1:j2]:
                    diffs.append(("", post_char))
            elif tag == "replace":
                assert i2 - i1 == j2 - j1, (pre_text[i1:i2], post_text[j1:j2])
                for pre_char, post_char in zip(pre_text[i1:i2], post_text[j1:j2]):
                    diffs.append((pre_char, post_char))
        return diffs
