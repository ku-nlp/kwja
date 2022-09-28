from enum import Enum

import torch


class PoolingStrategy(Enum):
    FIRST = "first"
    LAST = "last"
    MAX = "max"
    AVE = "average"


def pool_subwords(
    sequence_output: torch.Tensor,  # (b, seq, hid)
    subword_map: torch.Tensor,  # (b, word, seq)
    strategy: PoolingStrategy,
) -> torch.Tensor:  # (b, word, hid)
    """
    Convert each subword into a word unit by pooling the hidden states of each subword according to subword_map
    """
    batch_size, word_len, subword_len = subword_map.size()
    device = sequence_output.device
    if strategy == PoolingStrategy.FIRST:
        arrange = torch.arange(subword_len, 0, -1, device=device).view(1, 1, subword_len)  # (1, 1, seq)
        indices = torch.argmax(subword_map * arrange, dim=2)  # (b, word)
        # (b) -> (b, 1) -> (b, word)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(dim=1).expand(batch_size, word_len)
        return sequence_output[batch_indices, indices]  # (b, word, hid)
    if strategy == PoolingStrategy.LAST:
        arrange = torch.arange(0, subword_len, device=device).view(1, 1, subword_len)  # (1, 1, seq)
        indices = torch.argmax(subword_map * arrange, dim=2)  # (b, word)
        # (b) -> (b, 1) -> (b, word)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(dim=1).expand(batch_size, word_len)
        return sequence_output[batch_indices, indices]  # (b, word, hid)
    if strategy == PoolingStrategy.MAX:
        mask = ~subword_map * -1024.0  # (b, word, seq)
        word_sequence_output = torch.einsum("bsh,bws->bwsh", sequence_output, subword_map)  # (b, word, seq, hid)
        return torch.amax(word_sequence_output + mask.unsqueeze(dim=3), dim=2)  # (b, word, hid)
    if strategy == PoolingStrategy.AVE:
        subword_map = subword_map.float()
        conv_matrix = subword_map / (subword_map.sum(dim=2, keepdims=True) + 1e-6)  # (b, word, seq)
        return torch.bmm(conv_matrix, sequence_output)  # (b, word, hid)
