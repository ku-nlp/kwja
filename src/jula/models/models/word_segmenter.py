import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from jula.models.models.char_encoder import CharEncoder
from jula.utils.utils import SEG_LABEL2INDEX


class WordSegmenter(nn.Module):
    def __init__(self, hparams: DictConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.hparams = hparams

        self.char_encoder: CharEncoder = CharEncoder(
            hparams=hparams, tokenizer=tokenizer
        )

        self.hidden_size: int = self.char_encoder.pretrained_model.config.hidden_size
        self.seg_label_num: int = len(SEG_LABEL2INDEX)
        self.seg_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.char_encoder.pretrained_model.config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.seg_label_num),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.char_encoder(inputs)  # (b, seq_len, h)

        model_output: dict[str, torch.Tensor] = dict()
        seg_logits = self.seg_layers(output)  # (b, seq_len, seg_label_num)
        model_output["logits"] = seg_logits
        if "seg_labels" in inputs:
            seg_loss = F.cross_entropy(
                input=seg_logits.reshape(-1, self.seg_label_num),
                target=inputs["seg_labels"].view(-1),
                ignore_index=SEG_LABEL2INDEX["PAD"],
            )
            model_output["loss"] = seg_loss

        return model_output
