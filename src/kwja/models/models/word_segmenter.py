import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PretrainedConfig

from kwja.utils.constants import IGNORE_INDEX, SEG_TYPES


class WordSegmenter(nn.Module):
    def __init__(self, hparams: DictConfig, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        self.hparams = hparams

        self.hidden_size: int = pretrained_model_config.hidden_size
        self.num_labels: int = len(SEG_TYPES)
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(pretrained_model_config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.num_labels),
        )

    def forward(self, encoder_output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = dict()
        logits = self.cls(encoder_output)  # (b, seq_len, seg_label_num)
        output["logits"] = logits
        if "seg_types" in inputs:
            seg_loss = F.cross_entropy(
                input=logits.reshape(-1, self.num_labels),
                target=inputs["seg_types"].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["loss"] = seg_loss
        return output
