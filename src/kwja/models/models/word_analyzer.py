import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from kwja.utils.constants import CONJFORM_TYPES, CONJTYPE_TYPES, IGNORE_INDEX, POS_TYPES, SUBPOS_TYPES


class WordAnalyzer(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        hidden_size: int = pretrained_model_config.hidden_size
        self.num_pos_labels: int = len(POS_TYPES)
        self.num_subpos_labels: int = len(SUBPOS_TYPES)
        self.num_conjtype_labels: int = len(CONJTYPE_TYPES)
        self.num_conjform_labels: int = len(CONJFORM_TYPES)
        self.base_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(pretrained_model_config.hidden_dropout_prob),
        )
        self.pos_cls = nn.Linear(hidden_size, self.num_pos_labels)
        self.subpos_cls = nn.Linear(hidden_size, self.num_subpos_labels)
        self.conjtype_cls = nn.Linear(hidden_size, self.num_conjtype_labels)
        self.conjform_cls = nn.Linear(hidden_size, self.num_conjform_labels)

    def forward(self, encoder_output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = dict()
        base_output = self.base_cls(encoder_output)  # (b, seq_len, h)
        pos_logits = self.pos_cls(base_output)  # (b, seq_len, pos_label_num)
        subpos_logits = self.subpos_cls(base_output)  # (b, seq_len, subpos_label_num)
        conjtype_logits = self.conjtype_cls(base_output)  # (b, seq_len, conjtype_label_num)
        conjform_logits = self.conjform_cls(base_output)  # (b, seq_len, conjform_label_num)
        output["pos_logits"] = pos_logits
        output["subpos_logits"] = subpos_logits
        output["conjtype_logits"] = conjtype_logits
        output["conjform_logits"] = conjform_logits
        if "mrph_types" in inputs:
            output["pos_loss"] = F.cross_entropy(
                input=pos_logits.reshape(-1, self.num_pos_labels),
                target=inputs["mrph_types"][:, :, 0].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["subpos_loss"] = F.cross_entropy(
                input=subpos_logits.reshape(-1, self.num_subpos_labels),
                target=inputs["mrph_types"][:, :, 1].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["conjtype_loss"] = F.cross_entropy(
                input=conjtype_logits.reshape(-1, self.num_conjtype_labels),
                target=inputs["mrph_types"][:, :, 2].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["conjform_loss"] = F.cross_entropy(
                input=conjform_logits.reshape(-1, self.num_conjform_labels),
                target=inputs["mrph_types"][:, :, 3].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["loss"] = (
                output["pos_loss"] + output["subpos_loss"] + output["conjtype_loss"] + output["conjform_loss"]
            ) / 4
        return output
