import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel

from jula.models.models.pooling import PoolingStrategy, pool_subwords


class WordEncoder(nn.Module):
    def __init__(self, hparams: DictConfig, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        self.pretrained_model = AutoModel.from_pretrained(hparams.model.model_name_or_path, add_pooling_layer=False)
        self.pretrained_model.resize_token_embeddings(vocab_size)

    def forward(
        self, batch: dict[str, torch.Tensor], pooling_strategy: PoolingStrategy
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.pretrained_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled_outputs = pool_subwords(outputs.last_hidden_state, batch["subword_map"], pooling_strategy)
        return outputs.last_hidden_state, pooled_outputs  # both are in the shape of (batch_size, seq_len, hidden_size)
