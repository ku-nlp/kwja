import torch
import torch.nn as nn
from transformers import PretrainedConfig


class CohesionAnalyzer(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig, num_rels: int) -> None:
        super().__init__()

        self.dropout = nn.Dropout(0.0)

        self.num_rels = num_rels
        mid_hidden_size = pretrained_model_config.hidden_size

        self.l_src = nn.Linear(pretrained_model_config.hidden_size, mid_hidden_size * self.num_rels)
        self.l_tgt = nn.Linear(pretrained_model_config.hidden_size, mid_hidden_size * self.num_rels)
        self.out = nn.Linear(mid_hidden_size, 1, bias=False)

    def forward(
        self,
        pooled_outputs: torch.Tensor,  # (b, seq, hid)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (), (b, seq, seq)
        batch_size, sequence_len, _ = pooled_outputs.size()

        h_src = self.l_src(self.dropout(pooled_outputs))  # (b, seq, rel*hid)
        h_tgt = self.l_tgt(self.dropout(pooled_outputs))  # (b, seq, rel*hid)
        h_src = h_src.view(batch_size, sequence_len, self.num_rels, -1)  # (b, seq, rel, hid)
        h_tgt = h_tgt.view(batch_size, sequence_len, self.num_rels, -1)  # (b, seq, rel, hid)
        h = torch.tanh(self.dropout(h_src.unsqueeze(2) + h_tgt.unsqueeze(1)))  # (b, seq, seq, rel, hid)
        # -> (b, seq, seq, rel, 1) -> (b, seq, seq, rel) -> (b, rel, seq, seq)
        output = self.out(h).squeeze(-1).permute(0, 3, 1, 2).contiguous()

        return output


def cohesion_cross_entropy_loss(
    output: torch.Tensor,  # (b, rel, seq, seq)
    target: torch.Tensor,  # (b, rel, seq, seq)
    mask: torch.Tensor,  # (b, rel, seq, seq)
) -> torch.Tensor:  # ()
    eps = 1e-6
    output += (~mask).float() * -1024.0
    log_softmax = torch.log_softmax(output, dim=3)  # (b, rel, seq, seq)
    # reduce using masked mean (target ⊆ mask)
    return torch.sum(-log_softmax * target).div(torch.sum(target) + eps)
