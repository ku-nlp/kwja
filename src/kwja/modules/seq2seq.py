import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.generation import GenerationMixin, LogitsProcessorList

from kwja.modules.base import BaseModule
from kwja.modules.components.logits_processor import (
    SurfForcedDecodingLogitsProcessor,
    get_char2token_items,
    get_reading_candidate_token_ids,
)

if TYPE_CHECKING:
    from kwja.metrics import Seq2SeqModuleMetric
elif os.environ.get("KWJA_CLI_MODE") == "1":
    from kwja.modules.base import DummyModuleMetric as Seq2SeqModuleMetric  # dummy class for faster loading
else:
    from kwja.metrics import Seq2SeqModuleMetric


class Seq2SeqModule(BaseModule[Seq2SeqModuleMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams, Seq2SeqModuleMetric(hparams.max_src_length))

        self.tokenizer: PreTrainedTokenizerFast = hydra.utils.call(hparams.module.tokenizer)

        self.encoder_decoder: PreTrainedModel = hydra.utils.call(hparams.encoder.from_config)
        self._restore_untied_embeddings()
        # https://github.com/huggingface/transformers/issues/4875
        self.encoder_decoder.resize_token_embeddings(len(self.tokenizer.vocab))

        self.reading_candidate_token_ids = get_reading_candidate_token_ids(self.tokenizer)
        self.char2token_items = get_char2token_items(self.tokenizer)
        self.use_surf_forced_decoding: bool = getattr(hparams, "use_surf_forced_decoding", False)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.encoder_decoder = hydra.utils.call(self.hparams.encoder.from_pretrained)
            self._restore_untied_embeddings()
            # https://github.com/huggingface/transformers/issues/4875
            self.encoder_decoder.resize_token_embeddings(len(self.tokenizer.vocab))

    def _restore_untied_embeddings(self) -> None:
        if getattr(self.encoder_decoder.config, "scale_decoder_outputs", None) is False:
            self.encoder_decoder.config.tie_word_embeddings = False
            output_embeddings = self.encoder_decoder.get_output_embeddings()
            assert output_embeddings is not None  # noqa: S101
            self.encoder_decoder.set_output_embeddings(deepcopy(output_embeddings))

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        output = self.encoder_decoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["seq2seq_labels"]
        )
        return {"loss": output.loss, "logits": output.logits}

    def training_step(self, batch: Any) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
        self.log("train/loss", ret["loss"])
        return ret["loss"]

    def _get_metric_inputs(self, batch: Any) -> dict[str, torch.Tensor]:
        return {**self(batch), "example_ids": batch["example_ids"]}

    def predict_step(self, batch: Any) -> dict[str, Any]:
        assert isinstance(self.encoder_decoder, GenerationMixin)  # noqa: S101
        generations = self.encoder_decoder.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            logits_processor=(
                LogitsProcessorList(
                    [
                        SurfForcedDecodingLogitsProcessor(
                            batch_surfs=batch["surfs"],
                            num_beams=self.hparams.decoding.num_beams,
                            tokenizer=self.tokenizer,
                            char2token_items=self.char2token_items,
                            reading_candidate_token_ids=self.reading_candidate_token_ids,
                        ),
                    ]
                )
                if self.use_surf_forced_decoding
                else None
            ),
            **self.hparams.decoding,
        )
        if isinstance(generations, torch.Tensor):
            seq2seq_predictions = generations
        else:
            seq2seq_predictions = generations["sequences"]
        return {
            "example_ids": batch["example_ids"] if "example_ids" in batch else [],
            "seq2seq_predictions": seq2seq_predictions,
        }
