from typing import Optional

import torch


def unique(x: torch.Tensor, dim: Optional[int] = None):
    output, inverse_indices = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    arange = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
    inverse_indices, arange = inverse_indices.flip([0]), arange.flip([0])
    return inverse_indices.new_empty(output.size(0)).scatter_(0, inverse_indices, arange)
