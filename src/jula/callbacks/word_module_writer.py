import os

# from collections import defaultdict
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl

# import torch
from pytorch_lightning.callbacks import BasePredictionWriter

# from rhoknp import Morpheme
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.utils import (
    BASE_PHRASE_FEATURES,
    INDEX2CONJFORM_TYPE,
    INDEX2CONJTYPE_TYPE,
    INDEX2DEPENDENCY_TYPE,
    INDEX2POS_TYPE,
    INDEX2SUBPOS_TYPE,
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

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        results = []
        for prediction in predictions:
            for batch_pred in prediction:
                batch_texts = batch_pred["text"]
                # batch_pos_preds = torch.argmax(
                #     batch_pred["word_analysis_pos_logits"], dim=-1
                # )
                # batch_subpos_preds = torch.argmax(
                #     batch_pred["word_analysis_subpos_logits"], dim=-1
                # )
                # batch_conjtype_preds = torch.argmax(
                #     batch_pred["word_analysis_conjtype_logits"], dim=-1
                # )
                # batch_conjform_preds = torch.argmax(
                #     batch_pred["word_analysis_conjform_logits"], dim=-1
                # )
                # batch_phrase_analysis_preds = torch.where(
                #     batch_pred["phrase_analysis_logits"] >= 0.5, 1.0, 0.0
                # )
                # batch_dependency_preds = torch.argmax(
                #     batch_pred["dependency_logits"], dim=2
                # )
                # batch_dependency_type_preds = torch.argmax(
                #     batch_pred["dependency_type_logits"], dim=2
                # )
                for (
                    text,
                    # pos_preds,
                    # subpos_preds,
                    # conjtype_preds,
                    # conjform_preds,
                    # phrase_analysis_preds,
                    # dependency_preds,
                    # dependency_type_preds,
                ) in zip(
                    batch_texts,
                    # batch_pos_preds.tolist(),
                    # batch_subpos_preds.tolist(),
                    # batch_conjtype_preds.tolist(),
                    # batch_conjform_preds.tolist(),
                    # batch_phrase_analysis_preds.tolist(),
                    # batch_dependency_preds.tolist(),
                    # batch_dependency_type_preds.tolist(),
                ):
                    results.append(self.convert_predictions(text))
        if self.use_stdout:
            print("\n".join(results))
        else:
            with open(self.output_path, "w") as f:
                f.write("\n".join(results))

    @staticmethod
    def convert_word_analysis_pred(
        # morpheme: Morpheme,
        text: str,
        pos_pred: int,
        subpos_pred: int,
        conjtype_pred: int,
        conjform_pred: int,
    ):
        pred = (
            f"{INDEX2POS_TYPE[pos_pred]} {INDEX2SUBPOS_TYPE[subpos_pred]} "
            f"{INDEX2CONJTYPE_TYPE[conjtype_pred]} {INDEX2CONJFORM_TYPE[conjform_pred]}"
        )
        return pred
        # label = (
        #     f"{morpheme.pos} {morpheme.subpos} {morpheme.conjtype} {morpheme.conjform}"
        # )
        # return f"{pred}|{label}"

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
        system_head = pred - offset if pred != max_seq_len - 1 else -1
        system_deprel = INDEX2DEPENDENCY_TYPE[type_pred] if system_head >= 0 else "ROOT"
        if morpheme == morpheme.base_phrase.head:
            gold_head = morpheme.parent.index if morpheme.parent else -1
            gold_deprel = (
                morpheme.base_phrase.dep_type.value if morpheme.parent else "ROOT"
            )
            return f"{system_head}{system_deprel}|{gold_head}{gold_deprel}"
        else:
            gold_head = morpheme.base_phrase.head.index
            gold_deprel = "D"
            return f"{system_head}{system_deprel}|{gold_head}{gold_deprel}"

    def convert_predictions(
        self,
        text: str,
    ) -> str:
        words = text.split()
        return "\n".join(words)
        # id_, surf = morpheme.index, morpheme.surf
        # word_analysis_result = self.convert_word_analysis_pred(
        #     morpheme,
        #     pos_pred,
        #     subpos_pred,
        #     conjtype_pred,
        #     conjform_pred,
        # )
        # phrase_analysis_result = self.convert_phrase_analysis_pred(
        #     phrase_analysis_pred,
        #     base_phrase_feature,
        #     morpheme == morpheme.base_phrase.head,
        # )
        # dependency_parsing_result = self.convert_dependency_parsing_pred(
        #     morpheme, dependency_pred, dependency_type_pred, max_seq_len
        # )
        # return f"{id_}|{surf}|{word_analysis_result}|{phrase_analysis_result}|{dependency_parsing_result}"
