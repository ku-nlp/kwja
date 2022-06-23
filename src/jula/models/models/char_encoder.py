import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedTokenizer

from jula.utils.utils import ENE_TYPE_BIES


class CharEncoder(nn.Module):
    def __init__(self, hparams: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.hparams = hparams

        self.pretrained_model = AutoModel.from_pretrained(
            hparams.model.model_name_or_path, add_pooling_layer=False
        )
        self.pretrained_model.resize_token_embeddings(len(tokenizer))

        self.max_ene_num: int = self.hparams.dataset.max_ene_num
        if self.max_ene_num > 0:
            self.ene_pos_embed: nn.Embedding = nn.Embedding(
                len(ENE_TYPE_BIES), self.pretrained_model.config.hidden_size
            )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.max_ene_num > 0:
            batch_size, ene_num, seq_len = inputs["ene_ids"].size()
            ene_embed = self.ene_pos_embed(
                inputs["ene_ids"].view(-1, seq_len)
            )  # (b * ene_num, seq_len, h)
            ene_embed_sum = torch.sum(
                ene_embed.reshape(batch_size, ene_num, seq_len, -1), dim=1
            )  # (b, seq_len, h)
            inputs_embeds = self.pretrained_model.embeddings.word_embeddings(
                inputs["input_ids"]
            )  # (b, seq_len, h)
            outputs = self.pretrained_model(
                attention_mask=inputs["attention_mask"],
                inputs_embeds=inputs_embeds + ene_embed_sum,
            )
        else:
            outputs = self.pretrained_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        return outputs.last_hidden_state  # (b, seq_len, h)
