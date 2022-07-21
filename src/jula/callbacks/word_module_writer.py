import os
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    INDEX2CONJFORM_TYPE,
    INDEX2CONJTYPE_TYPE,
    INDEX2DEPENDENCY_TYPE,
    INDEX2POS_TYPE,
    INDEX2SUBPOS_TYPE,
    WORD_FEATURES,
)


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        tokenizer_kwargs: dict = None,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.use_stdout = use_stdout
        if self.use_stdout:
            self.output_path = ""
        else:
            self.output_path = f"{output_dir}/{pred_filename}.knp"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(self.output_path):
                os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs or {}, _convert_="partial"),
        )
        self.pad_token_id = self.tokenizer.pad_token_id

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
                batch_texts = batch_pred["text"]
                batch_pos_preds = torch.argmax(batch_pred["word_analysis_pos_logits"], dim=-1)
                batch_subpos_preds = torch.argmax(batch_pred["word_analysis_subpos_logits"], dim=-1)
                batch_conjtype_preds = torch.argmax(batch_pred["word_analysis_conjtype_logits"], dim=-1)
                batch_conjform_preds = torch.argmax(batch_pred["word_analysis_conjform_logits"], dim=-1)
                batch_word_feature_preds = batch_pred["word_feature_logits"].ge(0.5).long()
                batch_base_phrase_feature_preds = batch_pred["base_phrase_feature_logits"].ge(0.5).long()
                batch_dependency_preds = torch.topk(
                    batch_pred["dependency_logits"],
                    # pl_module.hparams.k,  # TODO: move to WordModuleWriter's config or argument
                    k=1,
                    dim=2,
                ).indices
                batch_dependency_type_preds = torch.argmax(batch_pred["dependency_type_logits"], dim=3)
                for values in zip(
                    batch_texts,
                    batch_pos_preds.tolist(),
                    batch_subpos_preds.tolist(),
                    batch_conjtype_preds.tolist(),
                    batch_conjform_preds.tolist(),
                    batch_word_feature_preds.tolist(),
                    batch_base_phrase_feature_preds.tolist(),
                    batch_dependency_preds.tolist(),
                    batch_dependency_type_preds.tolist(),
                ):
                    results.append(self.convert_predictions(*values))

        out = "\n".join(results)
        if self.use_stdout:
            print(out)
        else:
            with open(self.output_path, "w") as f:
                f.write(out)

    @staticmethod
    def convert_predictions(
        text: str,
        pos_preds: list[int],
        subpos_preds: list[int],
        conjtype_preds: list[int],
        conjform_preds: list[int],
        word_feature_preds: list[list[int]],
        base_phrase_feature_preds: list[list[int]],
        dependency_preds: list[list[int]],
        dependency_type_preds: list[list[int]],
    ) -> str:
        words = text.split()

        sequence_len = len(base_phrase_feature_preds)
        base_phrase_start_indices = {0}
        phrase_start_indices = {0}
        morpheme_index2base_phrase_index = {sequence_len - 1: -1}
        base_phrase_index2base_phrase_head = {}
        base_phrase_index = 0
        for i, word_feature_pred in enumerate(word_feature_preds[: len(words)]):
            morpheme_index2base_phrase_index[i] = base_phrase_index
            if word_feature_pred[WORD_FEATURES.index("基本句-主辞")] == 1:
                base_phrase_index2base_phrase_head[base_phrase_index] = i
            if word_feature_pred[WORD_FEATURES.index("基本句-区切")] == 1:
                base_phrase_start_indices.add(i + 1)
                base_phrase_index += 1
            if word_feature_pred[WORD_FEATURES.index("文節-区切")] == 1:
                phrase_start_indices.add(i + 1)

        results = []
        for i, word in enumerate(words):
            if i in phrase_start_indices:
                results.append("*")
            if i in base_phrase_start_indices:
                base_phrase_index = morpheme_index2base_phrase_index[i]
                base_phrase_head = base_phrase_index2base_phrase_head[base_phrase_index]
                parent_index = morpheme_index2base_phrase_index[dependency_preds[base_phrase_head][0]]
                dep_type = INDEX2DEPENDENCY_TYPE[dependency_type_preds[base_phrase_head][0]]
                features = "".join(
                    f"<{feature}>"
                    for feature, pred in zip(BASE_PHRASE_FEATURES, base_phrase_feature_preds[i])
                    if pred == 1
                )
                results.append(f"+ {parent_index}{dep_type.value} {features}")

            values = [
                word,
                INDEX2POS_TYPE[pos_preds[i]],
                INDEX2SUBPOS_TYPE[subpos_preds[i]],
                INDEX2CONJTYPE_TYPE[conjtype_preds[i]],
                INDEX2CONJFORM_TYPE[conjform_preds[i]],
            ]
            if word_feature_preds[i][WORD_FEATURES.index("基本句-主辞")] == 1:
                values.append("<基本句-主辞>")
            results.append(" ".join(values))
        results.append("EOS")
        return "\n".join(results)
