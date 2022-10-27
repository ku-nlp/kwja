import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.constants import TOKEN2TYPO_OPN


class TypoModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        extended_vocab_path: str,
        confidence_threshold: float = 0.0,
        pred_filename: str = "predict",
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        tokenizer_kwargs: DictConfig = None,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")

        self.destination = Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.txt")
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            if self.destination.exists():
                os.remove(str(self.destination))

        self.confidence_threshold = confidence_threshold

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )

        self.opn2id, self.id2opn = self.get_opn_dict(path=Path(extended_vocab_path))

    def get_opn_dict(self, path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
        opn2id: Dict[str, int] = self.tokenizer.get_vocab()
        id2opn: Dict[int, str] = {idx: opn for opn, idx in opn2id.items()}
        with path.open(mode="r") as f:
            for line in f:
                opn = str(line.strip())
                opn2id[opn] = len(opn2id)
                id2opn[len(id2opn)] = opn
        return opn2id, id2opn

    def convert_id2opn(self, opn_ids_list: List[List[int]], opn_prefix: str) -> List[List[str]]:
        opns_list: List[List[str]] = []
        for opn_ids in opn_ids_list:
            opns: List[str] = []
            for opn_id in opn_ids:
                opn: str = self.id2opn[opn_id]
                opns.append(TOKEN2TYPO_OPN.get(opn, f"{opn_prefix}:{opn}"))
            opns_list.append(opns)
        return opns_list

    @staticmethod
    def apply_opn(pre_text: str, kdrs: List[str], inss: List[str]) -> str:
        post_text = ""
        assert len(pre_text) + 1 == len(kdrs) + 1 == len(inss)
        for char_idx, char in enumerate(pre_text):
            # insert
            if inss[char_idx] != "_":
                post_text += inss[char_idx][2:]  # remove prefix "I:"
            # keep, delete, replace
            if kdrs[char_idx] == "K":
                post_text += char
            elif kdrs[char_idx] == "D":
                pass
            elif kdrs[char_idx].startswith("R:"):
                post_text += kdrs[char_idx][2:]  # remove prefix "R:"
            else:
                raise ValueError("unsupported operation!")
        if inss[-1] != "_":
            post_text += inss[-1][2:]  # remove prefix "I:"
        return post_text

    def get_opn_ids_list(
        self, batch_values: torch.Tensor, batch_indices: torch.Tensor, opn_prefix: str
    ) -> List[List[int]]:
        # Do not edit if the operation probability (replace, delete, and insert) is less than "confidence_threshold"
        opn_ids_list: List[List[int]] = []
        for values, indices in zip(batch_values.tolist(), batch_indices.tolist()):
            opn_ids: List[int] = []
            for value, index in zip(values, indices):
                if opn_prefix == "R" and value < self.confidence_threshold:
                    opn_ids.append(self.opn2id["<k>"])
                elif opn_prefix == "I" and value < self.confidence_threshold:
                    opn_ids.append(self.opn2id["<_>"])
                else:
                    opn_ids.append(index)
            opn_ids_list.append(opn_ids)
        return opn_ids_list

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        kdr_preds: List[List[str]] = self.convert_id2opn(
            opn_ids_list=self.get_opn_ids_list(
                batch_values=prediction["kdr_values"],
                batch_indices=prediction["kdr_indices"],
                opn_prefix="R",
            ),
            opn_prefix="R",
        )
        ins_preds: List[List[str]] = self.convert_id2opn(
            opn_ids_list=self.get_opn_ids_list(
                batch_values=prediction["ins_values"],
                batch_indices=prediction["ins_indices"],
                opn_prefix="I",
            ),
            opn_prefix="I",
        )
        results = []
        for idx in range(len(prediction["kdr_values"])):
            seq_len: int = len(prediction["texts"][idx])
            # the prediction of the dummy token (= "<dummy>") at the end of the input is used for insertion only.
            results.append(
                self.apply_opn(
                    pre_text=prediction["texts"][idx],
                    kdrs=kdr_preds[idx][:seq_len],
                    inss=ins_preds[idx][: seq_len + 1],
                )
            )

        output_string = "".join(result + "\nEOD\n" for result in results)  # Append EOD to the end of each result
        if isinstance(self.destination, Path):
            with self.destination.open("a") as f:
                f.write(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        pass
