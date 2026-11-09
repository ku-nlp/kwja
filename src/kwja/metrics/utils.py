import torch


def first_occurrence_indices(x: torch.Tensor) -> torch.Tensor:
    """Indices of the first occurrence of each distinct value in a 1-D tensor, in order of position."""
    unique_values, inverse = torch.unique(x, return_inverse=True)
    positions = torch.arange(x.numel(), device=x.device)
    first = torch.full((unique_values.numel(),), x.numel(), dtype=positions.dtype, device=x.device)
    return first.scatter_reduce(0, inverse, positions, reduce="amin").sort().values
