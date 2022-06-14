import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedTokenizerBase


class WordEncoder(nn.Module):
    def __init__(self, hparams: DictConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.hparams = hparams
        self.pretrained_model = AutoModel.from_pretrained(
            hparams.model.model_name_or_path, add_pooling_layer=False
        )
        self.pretrained_model.resize_token_embeddings(len(tokenizer))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.pretrained_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs.last_hidden_state  # (b, seq_len, h)
