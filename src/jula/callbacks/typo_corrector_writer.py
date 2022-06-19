import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

from jula.evaluators.typo_corrector import TypoCorrectorMetric
from jula.utils.utils import TOKEN2TYPO_OPN, TYPO_OPN2TOKEN


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

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial"),
        )
        self.predicts: dict[int, Any] = dict()
        self.metrics: TypoCorrectorMetric = TypoCorrectorMetric()

        self.opn2id, self.id2opn = self.get_opn_dict(path=Path(extended_vocab_path))

    def get_opn_dict(self, path: Path) -> tuple[dict[str, int], dict[int, str]]:
        opn2id: dict[str, int] = self.tokenizer.get_vocab()
        id2opn: dict[int, str] = {idx: opn for opn, idx in opn2id.items()}
        with path.open(mode="r") as f:
            for line in f:
                opn = str(line.strip())
                opn2id[opn] = len(opn2id)
                id2opn[len(id2opn)] = opn
        return opn2id, id2opn

    def get_opn(
        self,
        pred_ids_list: list[list[int]],
        label_ids_list: list[list[int]],
        opn_prefix: str,
    ) -> tuple[list[list[str]], list[list[str]]]:
        preds_list: list[list[str]] = []
        labels_list: list[list[str]] = []
        for pred_ids, label_ids in zip(pred_ids_list, label_ids_list):
            preds: list[str] = []
            labels: list[str] = []
            for pred_id, label_id in zip(pred_ids, label_ids):
                if label_id == self.tokenizer.pad_token_id:
                    continue

                if self.id2opn[pred_id] in TYPO_OPN2TOKEN.values():
                    preds.append(TOKEN2TYPO_OPN[self.id2opn[pred_id]])
                else:
                    preds.append(f"{opn_prefix}:{self.id2opn[pred_id]}")

                if self.id2opn[label_id] in TYPO_OPN2TOKEN.values():
                    labels.append(TOKEN2TYPO_OPN[self.id2opn[label_id]])
                else:
                    labels.append(f"{opn_prefix}:{self.id2opn[label_id]}")
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
                kdr_preds, kdr_labels = self.get_opn(
                    pred_ids_list=torch.argmax(
                        batch_pred["kdr_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["kdr_labels"].tolist(),
                    opn_prefix="R",
                )
                ins_preds, ins_labels = self.get_opn(
                    pred_ids_list=torch.argmax(
                        batch_pred["ins_logits"], dim=-1
                    ).tolist(),
                    label_ids_list=batch_pred["ins_labels"].tolist(),
                    opn_prefix="I",
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
