import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.typo_corrector_metrics import TypoCorrectorMetrics
from jula.utils.utils import TYPO_OPS2TOKEN


class TypoCorrectorWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        extended_vocab_path: str,
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
            **hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial"),
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.predicts = dict()
        self.metrics = TypoCorrectorMetrics()

        self.ops2id, self.id2ops = self.get_ops_dict(path=Path(extended_vocab_path))

    def get_ops_dict(self, path: Path) -> tuple[dict[str, int], dict[int, str]]:
        ops2id: dict[str, int] = self.tokenizer.get_vocab()
        id2ops: dict[int, str] = {idx: ops for ops, idx in ops2id.items()}
        with path.open(mode="r") as f:
            for line in f:
                ops = str(line.strip())
                ops2id[ops] = len(ops2id)
        return ops2id, id2ops

    def get_ops(
        self,
        pred_ids_list: list[list[int]],
        label_ids_list: list[list[int]],
        ops_prefix: str,
    ) -> tuple[list[list[int]], list[list[int]]]:
        preds_list: list[list[str]] = []
        labels_list: list[list[str]] = []
        for pred_ids, label_ids in zip(pred_ids_list, label_ids_list):
            preds: list[str] = []
            labels: list[str] = []
            for pred_id, label_id in zip(pred_ids, label_ids):
                if label_id == self.tokenizer.pad_token_id:
                    continue

                if self.id2ops[pred_id] in TYPO_OPS2TOKEN.values():
                    preds.append(self.id2ops[pred_id])
                else:
                    preds.append(f"{ops_prefix}:{self.id2ops[pred_id]}")

                if self.id2ops[label_id] in TYPO_OPS2TOKEN.values():
                    labels.append(self.id2ops[label_id])
                else:
                    labels.append(f"{ops_prefix}:{self.id2ops[label_id]}")
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
                kdr_preds, kdr_labels = self.get_ops(
                    pred_ids_list=torch.argmax(
                        batch_pred["kdr_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["kdr_labels"].tolist(),
                    ops_prefix="R",
                )
                ins_preds, ins_labels = self.get_ops(
                    pred_ids_list=torch.argmax(
                        batch_pred["ins_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["ins_labels"].tolist(),
                    ops_prefix="I",
                )
                for idx in range(len(batch_pred["input_ids"])):
                    self.predicts[example_id] = dict(
                        input_ids=self.tokenizer.decode(
                            [x for x in batch_pred["input_ids"][idx]][:-1],
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
