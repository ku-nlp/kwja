import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

from jula.utils.constants import ENE_TYPE_BIES


class CharEncoder(nn.Module):
    def __init__(self, hparams: DictConfig, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams

        self.pretrained_model = AutoModel.from_pretrained(hparams.model.model_name_or_path, add_pooling_layer=False)
        self.pretrained_model.resize_token_embeddings(vocab_size)
        self.word_embed = self.pretrained_model.embeddings.word_embeddings

        self.max_ene_num: int = self.hparams.dataset.get("max_ene_num", 0)
        if self.max_ene_num > 0:
            self.add_ene_embed_before_encoder: bool = hparams.add_ene_embed_before_encoder
            self.ene_embed: nn.Embedding = nn.Embedding(
                len(ENE_TYPE_BIES),
                self.pretrained_model.config.hidden_size,
                padding_idx=ENE_TYPE_BIES.index("PAD"),
            )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.max_ene_num > 0:
            batch_size, ene_num, seq_len = inputs["ene_ids"].size()
            ene_embed = self.ene_embed(inputs["ene_ids"].view(-1, seq_len))  # (b * ene_num, seq_len, h)
            ene_embed_sum = torch.sum(ene_embed.reshape(batch_size, ene_num, seq_len, -1), dim=1)  # (b, seq_len, h)

            if self.add_ene_embed_before_encoder:
                position_ids = create_position_ids_from_input_ids(
                    inputs["input_ids"],
                    padding_idx=self.pretrained_model.config.pad_token_id,
                )
                inputs_embeds = self.word_embed(inputs["input_ids"])  # (b, seq_len, h)
                outputs = self.pretrained_model(
                    attention_mask=inputs["attention_mask"],
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds + ene_embed_sum,
                )
                encoder_output = outputs.last_hidden_state
            else:
                outputs = self.pretrained_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                encoder_output = outputs.last_hidden_state + ene_embed_sum
        else:
            outputs = self.pretrained_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            encoder_output = outputs.last_hidden_state

        return encoder_output  # (b, seq_len, h)
