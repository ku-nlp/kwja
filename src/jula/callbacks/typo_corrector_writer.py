import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.typo_corrector_metrics import TypoCorrectorMetrics


class TypoCorrectorWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        insert_vocab_path: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        tokenizer_kwargs: dict = None,
    ) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = f"{output_dir}/{pred_filename}.json"
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.predicts = dict()
        self.metrics = TypoCorrectorMetrics()

        self.char2id, self.id2char = self.get_vocab(path=Path(insert_vocab_path))

    def get_vocab(self, path: Path) -> tuple[dict[str, int], dict[int, str]]:
        char2id = self.tokenizer.get_vocab()
        id2char: dict[int, str] = {
            idx: char for char, idx in self.tokenizer.get_vocab().items()
        }
        with path.open(mode="r") as f:
            for line in f:
                char = str(line.strip())
                id2char[len(id2char)] = char
                char2id[char] = len(char2id)
        return char2id, id2char

    def get_kdrs(self, pred_ids_list: list[list[int]], label_ids_list: list[list[int]]):
        preds_list: list[list[str]] = []
        labels_list: list[list[str]] = []
        for pred_ids, label_ids in zip(pred_ids_list, label_ids_list):
            preds: list[str] = []
            labels: list[str] = []
            for pred_id, label_id in zip(pred_ids, label_ids):
                if label_id == self.tokenizer.pad_token_id:
                    continue
                pred = self.id2char[pred_id]
                if pred == "<0x01>":
                    preds.append("K")
                elif pred == "<0x02>":
                    preds.append("D")
                else:
                    preds.append(f"R:{pred}")
                label = self.id2char[label_id]
                if label == "<0x01>":
                    labels.append("K")
                elif label == "<0x02>":
                    labels.append("D")
                else:
                    labels.append(f"R:{label}")
            preds_list.append(preds)
            labels_list.append(labels)
        return preds_list, labels_list

    def get_inss(self, pred_ids_list: list[list[int]], label_ids_list: list[list[int]]):
        preds_list: list[list[str]] = []
        labels_list: list[list[str]] = []
        for pred_ids, label_ids in zip(pred_ids_list, label_ids_list):
            preds: list[str] = []
            labels: list[str] = []
            for pred_id, label_id in zip(pred_ids, label_ids):
                if label_id == self.tokenizer.pad_token_id:
                    continue
                pred = self.id2char[pred_id]
                if pred == "<0x03>":
                    preds.append("_")
                else:
                    preds.append(f"I:{pred}")
                label = self.id2char[label_id]
                if label == "<0x03>":
                    labels.append("_")
                else:
                    labels.append(f"I:{label}")
            preds_list.append(preds)
            labels_list.append(labels)
        return preds_list, labels_list

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        example_id = 0
        for prediction in predictions:
            for batch_pred in prediction:
                kdr_preds, kdr_labels = self.get_kdrs(
                    pred_ids_list=torch.argmax(
                        batch_pred["kdr_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["kdr_labels"].tolist(),
                )
                ins_preds, ins_labels = self.get_inss(
                    pred_ids_list=torch.argmax(
                        batch_pred["ins_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["ins_labels"].tolist(),
                )
                for idx in range(len(batch_pred["input_ids"])):
                    self.predicts[example_id] = dict(
                        input_ids=self.tokenizer.decode(
                            [
                                x
                                for x in batch_pred["input_ids"][idx]
                                if x != self.char2id["<0x00>"]
                            ],
                            skip_special_tokens=True,
                        ),
                        kdr_preds=kdr_preds[idx],
                        kdr_labels=kdr_labels[idx],
                        ins_preds=ins_preds[idx],
                        ins_labels=ins_labels[idx],
                    )
                    example_id += 1

        with open(self.output_path, "w") as f:
            json.dump(self.predicts, f, ensure_ascii=False, indent=2)
