import json
import os
from collections import defaultdict
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from rhoknp import Morpheme
from rhoknp.units.utils import DepType
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.utils import (
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
            self.output_path = f"{output_dir}/{pred_filename}.json"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(self.output_path):
                os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial"),
        )
        self.pad_token_id = self.tokenizer.pad_token_id

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
        results = defaultdict(list)
        for dataloader_idx, prediction_step_outputs in enumerate(predictions):
            corpus = pl_module.test_corpora[dataloader_idx]
            dataset = trainer.datamodule.test_datasets[corpus]
            for prediction_step_output in prediction_step_outputs:
                batch_pos_preds = torch.argmax(
                    prediction_step_output["word_analysis_pos_logits"], dim=-1
                )
                batch_subpos_preds = torch.argmax(
                    prediction_step_output["word_analysis_subpos_logits"], dim=-1
                )
                batch_conjtype_preds = torch.argmax(
                    prediction_step_output["word_analysis_conjtype_logits"], dim=-1
                )
                batch_conjform_preds = torch.argmax(
                    prediction_step_output["word_analysis_conjform_logits"], dim=-1
                )
                batch_word_feature_predictions = (
                    prediction_step_output["word_feature_logits"].ge(0.5).long()
                )
                batch_base_phrase_feature_predictions = (
                    prediction_step_output["base_phrase_feature_logits"].ge(0.5).long()
                )
                batch_dependency_predictions = torch.topk(
                    prediction_step_output["dependency_logits"],
                    pl_module.hparams.k,
                    dim=2,
                ).indices
                batch_dependency_type_predictions = torch.argmax(
                    prediction_step_output["dependency_type_logits"], dim=3
                )
                for (
                    example_id,
                    pos_preds,
                    subpos_preds,
                    conjtype_preds,
                    conjform_preds,
                    word_feature_predictions,
                    base_phrase_feature_predictions,
                    dependency_predictions,
                    dependency_type_predictions,
                ) in zip(
                    prediction_step_output["example_ids"],
                    batch_pos_preds.tolist(),
                    batch_subpos_preds.tolist(),
                    batch_conjtype_preds.tolist(),
                    batch_conjform_preds.tolist(),
                    batch_word_feature_predictions.tolist(),
                    batch_base_phrase_feature_predictions.tolist(),
                    batch_dependency_predictions.tolist(),
                    batch_dependency_type_predictions.tolist(),
                ):
                    document = dataset.documents[example_id]
                    results[corpus].append(
                        [
                            self.convert_predictions(
                                values, len(dependency_predictions)
                            )
                            for values in zip(
                                document.morphemes,
                                pos_preds,
                                subpos_preds,
                                conjtype_preds,
                                conjform_preds,
                                word_feature_predictions,
                                base_phrase_feature_predictions,
                                dependency_predictions,
                                dependency_type_predictions,
                            )
                        ]
                    )

        if self.use_stdout:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            with open(self.output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    @staticmethod
    def convert_word_analysis_pred(
        morpheme: Morpheme,
        pos_pred: int,
        subpos_pred: int,
        conjtype_pred: int,
        conjform_pred: int,
    ):
        pred = (
            f"{INDEX2POS_TYPE[pos_pred]} {INDEX2SUBPOS_TYPE[subpos_pred]} "
            f"{INDEX2CONJTYPE_TYPE[conjtype_pred]} {INDEX2CONJFORM_TYPE[conjform_pred]}"
        )
        label = (
            f"{morpheme.pos} {morpheme.subpos} {morpheme.conjtype} {morpheme.conjform}"
        )
        return f"{pred}|{label}"

    @staticmethod
    def convert_phrase_analysis_prediction(
        word_feature_prediction: list[int],
        base_phrase_feature_prediction: list[int],
        is_base_phrase_head: bool,
    ):
        word_features = "".join(
            f"<{feature}>"
            for feature, pred in zip(WORD_FEATURES, word_feature_prediction)
            if pred == 1
        )
        if not is_base_phrase_head:
            return f"{word_features}|"
        else:
            base_phrase_features = "".join(
                f"<{feature}>"
                for feature, pred in zip(
                    BASE_PHRASE_FEATURES, base_phrase_feature_prediction
                )
                if pred == 1
            )
            return f"{word_features}|{base_phrase_features}"

    @staticmethod
    def convert_dependency_parsing_prediction(
        morpheme: Morpheme,
        topk_heads: list[int],
        topk_dependency_types: list[int],
        sequence_len: int,
    ):
        head = topk_heads[0]
        dependency_type = topk_dependency_types[0]
        offset = min(morpheme.global_index for morpheme in morpheme.sentence.morphemes)
        if head == sequence_len - 1:
            system_head = -1
            system_deprel = DepType.DEPENDENCY
        else:
            system_head = head - offset
            system_deprel = INDEX2DEPENDENCY_TYPE[dependency_type]
        return f"{system_head}{system_deprel.value}"

    def convert_predictions(self, values, sequence_len: int):
        (
            morpheme,
            pos_pred,
            subpos_pred,
            conjtype_pred,
            conjform_pred,
            word_feature_prediction,
            base_phrase_feature_prediction,
            dependency_prediction,
            dependency_type_prediction,
        ) = values
        id_, surf = morpheme.index, morpheme.surf
        word_analysis_result = self.convert_word_analysis_pred(
            morpheme,
            pos_pred,
            subpos_pred,
            conjtype_pred,
            conjform_pred,
        )
        phrase_analysis_result = self.convert_phrase_analysis_prediction(
            word_feature_prediction,
            base_phrase_feature_prediction,
            morpheme == morpheme.base_phrase.head,
        )
        dependency_parsing_result = self.convert_dependency_parsing_prediction(
            morpheme, dependency_prediction, dependency_type_prediction, sequence_len
        )
        return f"{id_}|{surf}|{word_analysis_result}|{phrase_analysis_result}|{dependency_parsing_result}"
