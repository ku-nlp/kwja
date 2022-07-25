import os
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

from jula.utils.constants import TOKEN2TYPO_OPN, TYPO_OPN2TOKEN


class TypoCorrectorWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        extended_vocab_path: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        tokenizer_kwargs: dict = None,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.use_stdout = use_stdout
        if self.use_stdout:
            self.output_path = ""
        else:
            self.output_path = f"{output_dir}/{pred_filename}.txt"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(self.output_path):
                os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs or {}, _convert_="partial"),
        )

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

    def convert_id2opn(self, opn_ids_list: list[list[int]], opn_prefix: str) -> list[list[str]]:
        opns_list: list[list[str]] = []
        for opn_ids in opn_ids_list:
            opns: list[str] = []
            for opn_id in opn_ids:
                if self.id2opn[opn_id] in TYPO_OPN2TOKEN.values():
                    opns.append(TOKEN2TYPO_OPN[self.id2opn[opn_id]])
                else:
                    opns.append(f"{opn_prefix}:{self.id2opn[opn_id]}")
            opns_list.append(opns)
        return opns_list

    @staticmethod
    def apply_opn(pre_text: str, kdrs: list[str], inss: list[str]) -> str:
        post_text = ""
        assert len(pre_text) + 1 == len(kdrs) + 1 == len(inss)
        for char_idx, char in enumerate(pre_text):
            # insert
            if inss[char_idx] != "_":
                post_text += inss[char_idx].removeprefix("I:")
            # keep, delete, replace
            if kdrs[char_idx] == "K":
                post_text += char
            elif kdrs[char_idx] == "D":
                pass
            elif kdrs[char_idx].startswith("R:"):
                post_text += kdrs[char_idx].removeprefix("R:")
            else:
                raise ValueError("unsupported operation!")
        if inss[-1] != "_":
            post_text += inss[-1].removeprefix("I:")
        return post_text

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        results = []
        for prediction in predictions:
            for batch_pred in prediction:
                # the prediction of the first token (= [CLS]) is excluded.
                kdr_preds: list[list[str]] = self.convert_id2opn(
                    opn_ids_list=torch.argmax(batch_pred["kdr_logits"][:, 1:], dim=-1).tolist(),
                    opn_prefix="R",
                )
                ins_preds: list[list[str]] = self.convert_id2opn(
                    opn_ids_list=torch.argmax(batch_pred["ins_logits"][:, 1:], dim=-1).tolist(),
                    opn_prefix="I",
                )
                result = []
                for idx in range(len(batch_pred["kdr_logits"])):
                    seq_len: int = len(batch_pred["texts"][idx])
                    # the prediction of the dummy token (= "<dummy>") at the end of the input is used for insertion only.
                    result.append(
                        self.apply_opn(
                            pre_text=batch_pred["texts"][idx],
                            kdrs=kdr_preds[idx][:seq_len],
                            inss=ins_preds[idx][: seq_len + 1],
                        )
                    )
                results.append("\n".join(result))

        out = "\n".join(results)
        if self.use_stdout:
            print(out)
        else:
            with open(self.output_path, "w") as f:
                f.write(out)
