from typing import Dict, List, Tuple

import torch
from Levenshtein import opcodes

from kwja.datamodule.datasets.typo_dataset import TypoDataset
from kwja.utils.typo_module_writer import apply_edit_operations, convert_predictions_into_typo_corr_op_tags


class TypoModuleMetric:
    def __init__(self) -> None:
        self.confidence_thresholds = [0.0, 0.8, 0.9]
        self.example_id2texts: Dict[str, Tuple[str, List[str], str]] = {}

    def update(self, metric_args: Dict[str, torch.Tensor], dataset: TypoDataset) -> None:
        for (example_id, kdr_predictions, kdr_probabilities, ins_predictions, ins_probabilities) in zip(
            metric_args["example_ids"],
            metric_args["kdr_predictions"].tolist(),
            metric_args["kdr_probabilities"].tolist(),
            metric_args["ins_predictions"].tolist(),
            metric_args["ins_probabilities"].tolist(),
        ):
            example = dataset.examples[example_id]
            seq_len: int = len(example["pre_text"])
            if seq_len == 0:
                continue

            predicted_texts = []
            for confidence_threshold in self.confidence_thresholds:
                args = (confidence_threshold, dataset.token2token_id, dataset.token_id2token)
                kdr_tags = convert_predictions_into_typo_corr_op_tags(kdr_predictions, kdr_probabilities, "R", *args)
                ins_tags = convert_predictions_into_typo_corr_op_tags(ins_predictions, ins_probabilities, "I", *args)

                # the prediction of the first token (= [CLS]) is excluded.
                # the prediction of the dummy token at the end is used for insertion only.
                predicted_text = apply_edit_operations(
                    example["pre_text"], kdr_tags[1 : seq_len + 1], ins_tags[1 : seq_len + 2]
                )
                predicted_texts.append(predicted_text)
            self.example_id2texts[example_id] = (example["pre_text"], predicted_texts, example["post_text"])

    def compute(self) -> Dict[str, float]:
        tps = [0] * len(self.confidence_thresholds)
        fps = [0] * len(self.confidence_thresholds)
        fns = [0] * len(self.confidence_thresholds)
        for pre_text, predicted_texts, gold_text in self.example_id2texts.values():
            gold_diffs = self._get_diffs(pre_text, gold_text)
            for i, predicted_text in enumerate(predicted_texts):
                predicted_diffs = self._get_diffs(pre_text, predicted_text)
                intersection = []
                queue = [*gold_diffs]
                for predicted_diff in predicted_diffs:
                    if predicted_diff in queue:
                        intersection.append(predicted_text)
                        queue.remove(predicted_diff)
                assert (
                    len(predicted_diffs) - len(intersection) >= 0 and len(gold_diffs) - len(intersection) >= 0
                ), "invalid tp"
                tps[i] += len(intersection)
                fps[i] += len(predicted_diffs) - len(intersection)
                fns[i] += len(gold_diffs) - len(intersection)

        metrics: Dict[str, float] = {}
        for tp, fp, fn, confidence_threshold in zip(tps, fps, fns, self.confidence_thresholds):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if (precision + recall) == 0.0:
                f1 = 0.0
                f05 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            metrics.update(
                {
                    f"typo_correction_{confidence_threshold}_precision": precision,
                    f"typo_correction_{confidence_threshold}_recall": recall,
                    f"typo_correction_{confidence_threshold}_f1": f1,
                    f"typo_correction_{confidence_threshold}_f0.5": f05,
                }
            )
        return metrics

    def reset(self) -> None:
        self.example_id2texts.clear()

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
