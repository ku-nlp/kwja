from typing import Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import PreTrainedModel

from kwja.models.models.pooling import PoolingStrategy, pool_subwords


class WordEncoder(nn.Module):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.pretrained_model: PreTrainedModel = hydra.utils.call(hparams.encoder)
        if hasattr(hparams.dataset, "special_tokens"):
            self.pretrained_model.resize_token_embeddings(
                self.pretrained_model.config.vocab_size + len(hparams.dataset.special_tokens)
            )

    def forward(
        self, batch: Dict[str, torch.Tensor], pooling_strategy: PoolingStrategy
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.pretrained_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled_outputs = pool_subwords(outputs.last_hidden_state, batch["subword_map"], pooling_strategy)
        return outputs.last_hidden_state, pooled_outputs  # both are in the shape of (batch_size, seq_len, hidden_size)
