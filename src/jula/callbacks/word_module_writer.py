import json
import os
from collections import defaultdict
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.utils import BASE_PHRASE_FEATURES, INDEX2DEPENDENCY_TYPE


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
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
                batch_phrase_analysis_preds = torch.where(
                    prediction_step_output["phrase_analysis_logits"] >= 0.5, 1.0, 0.0
                )
                batch_dependency_preds = torch.argmax(
                    prediction_step_output["dependency_logits"], dim=2
                )
                batch_dependency_type_preds = torch.argmax(
                    prediction_step_output["dependency_type_logits"], dim=2
                )
                for (
                    document_id,
                    phrase_analysis_preds,
                    base_phrase_features,
                    dependency_preds,
                    dependency_type_preds,
                ) in zip(
                    prediction_step_output["document_ids"],
                    batch_phrase_analysis_preds.tolist(),
                    prediction_step_output["base_phrase_features"].tolist(),
                    batch_dependency_preds.tolist(),
                    batch_dependency_type_preds.tolist(),
                ):
                    document = dataset.documents[document_id]
                    results[corpus].append(
                        [
                            self.convert_predictions(values, len(dependency_preds))
                            for values in zip(
                                document.morphemes,
                                phrase_analysis_preds,
                                base_phrase_features,
                                dependency_preds,
                                dependency_type_preds,
                            )
                        ]
                    )

        with open(self.output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    @staticmethod
    def convert_phrase_analysis_pred(pred, label, head):
        pred, label = map(
            lambda x: " ".join(
                f"<{feature}>"
                for feature, element in zip(BASE_PHRASE_FEATURES, x)
                if element == 1.0
            ),
            [pred, label],
        )
        if not head:
            return f"|{label}"
        else:
            return f"{pred}|{label}"

    @staticmethod
    def convert_dependency_parsing_pred(morpheme, pred, type_pred, max_seq_len):
        offset = min(morpheme.global_index for morpheme in morpheme.sentence.morphemes)
        system_head = pred - offset if pred != max_seq_len - 1 else 0
        system_deprel = INDEX2DEPENDENCY_TYPE[type_pred] if system_head > 0 else "ROOT"
        if morpheme == morpheme.base_phrase.head:
            gold_head = morpheme.parent.index if morpheme.parent else 0
            gold_deprel = (
                morpheme.base_phrase.dep_type.value if morpheme.parent else "ROOT"
            )
            return f"{system_head}{system_deprel}|{gold_head}{gold_deprel}"
        else:
            gold_head = morpheme.base_phrase.head.index
            gold_deprel = "D"
            return f"{system_head}{system_deprel}|{gold_head}{gold_deprel}"

    def convert_predictions(self, values, max_seq_len):
        (
            morpheme,
            phrase_analysis_pred,
            base_phrase_feature,
            dependency_pred,
            dependency_type_pred,
        ) = values
        id_, surf = morpheme.index, morpheme.surf
        phrase_analysis_result = self.convert_phrase_analysis_pred(
            phrase_analysis_pred,
            base_phrase_feature,
            morpheme == morpheme.base_phrase.head,
        )
        dependency_parsing_result = self.convert_dependency_parsing_pred(
            morpheme, dependency_pred, dependency_type_pred, max_seq_len
        )
        return f"{id_}|{surf}|{phrase_analysis_result}|{dependency_parsing_result}"
