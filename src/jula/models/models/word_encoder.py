import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedTokenizerBase

from jula.models.models.pooling import PoolingStrategy, pool_subwords


class WordEncoder(nn.Module):
    def __init__(self, hparams: DictConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.hparams = hparams
        self.pretrained_model = AutoModel.from_pretrained(hparams.model.model_name_or_path, add_pooling_layer=False)
        self.pretrained_model.resize_token_embeddings(len(tokenizer))

    def forward(self, batch: dict[str, torch.Tensor], pooling_strategy: PoolingStrategy) -> torch.Tensor:
        outputs = self.pretrained_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled_outputs = pool_subwords(outputs.last_hidden_state, batch["subword_map"], pooling_strategy)
        return pooled_outputs  # (batch_size, seq_len, hidden_size)
