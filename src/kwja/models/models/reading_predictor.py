from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from kwja.utils.constants import IGNORE_INDEX
from kwja.utils.reading import get_reading2id


class ReadingPredictor(nn.Module):
    def __init__(self, reading_resource_path: str, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()

        self.reading_resource_path = Path(reading_resource_path)
        self.reading2id = get_reading2id(str(self.reading_resource_path / "vocab.txt"))

        hidden_size = pretrained_model_config.hidden_size
        hidden_dropout_prob = pretrained_model_config.hidden_dropout_prob
        self.reading_predictor_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    ("dropout", nn.Dropout(hidden_dropout_prob)),
                    ("fc", nn.Linear(hidden_size, len(self.reading2id))),
                ]
            )
        )

    def forward(self, pooled_outputs: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        reading_predictor_logits = self.reading_predictor_head(pooled_outputs)  # (b, seq, num_labels)
        outputs["logits"] = reading_predictor_logits
        if "reading_ids" in batch:
            loss = F.cross_entropy(
                input=reading_predictor_logits.view(-1, reading_predictor_logits.size(2)),
                target=batch["reading_ids"].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            outputs["loss"] = loss
        return outputs
