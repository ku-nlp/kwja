import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedModel


class CharEncoder(nn.Module):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams = hparams

        self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(
            hparams.model.model_name_or_path, add_pooling_layer=False
        )
        if hasattr(hparams.dataset, "special_tokens"):
            self.pretrained_model.resize_token_embeddings(
                self.pretrained_model.config.vocab_size + len(hparams.dataset.special_tokens)
            )
        self.word_embed = self.pretrained_model.embeddings.word_embeddings

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.pretrained_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs.last_hidden_state  # (b, seq_len, h)
