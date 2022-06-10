import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from jula.models.models.char_encoder import CharEncoder


class TypoCorrector(nn.Module):
    def __init__(self, hparams: DictConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

        self.char_encoder: CharEncoder = CharEncoder(
            hparams=hparams, tokenizer=tokenizer
        )

        self.hidden_size: int = self.char_encoder.pretrained_model.config.hidden_size
        self.kdr_label_num: int = len(tokenizer)
        self.kdr_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.char_encoder.pretrained_model.config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.kdr_label_num),
        )
        self.ins_label_num: int = len(tokenizer) + hparams.dataset.extended_vocab_size
        self.ins_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.char_encoder.pretrained_model.config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.ins_label_num),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.char_encoder(inputs)  # (b, seq_len, h)

        model_output: dict[str, torch.Tensor] = dict()
        kdr_logits = self.kdr_layers(output)  # (b, seq_len, kdr_label_num)
        model_output["kdr_logits"] = kdr_logits
        ins_logits = self.ins_layers(output)  # (b, seq_len, ins_label_num)
        model_output["ins_logits"] = ins_logits
        if "kdr_labels" in inputs and "ins_labels" in inputs:
            kdr_loss = F.cross_entropy(
                input=kdr_logits.reshape(-1, self.kdr_label_num),
                target=inputs["kdr_labels"].view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            model_output["kdr_loss"] = kdr_loss
            ins_loss = F.cross_entropy(
                input=ins_logits.reshape(-1, self.ins_label_num),
                target=inputs["ins_labels"].view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            model_output["ins_loss"] = ins_loss
            model_output["loss"] = (kdr_loss + ins_loss) / 2

        return model_output
