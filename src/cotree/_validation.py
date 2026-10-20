"""Shared validation for trajectory datasets and cell-level splits."""

import numpy as np
import torch


def validate_series(series, names):
    """Convert one cell's modalities to finite, equally long float tensors."""
    result = {}
    length = None
    for name in names:
        if name not in series:
            raise ValueError(f"Missing modality {name!r}.")
        value = series[name].to(dtype=torch.float32) if isinstance(series[name], torch.Tensor) else torch.tensor(np.array(series[name], copy=True), dtype=torch.float32)
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] == 0:
            raise ValueError(f"{name}: expected (time >= 2, features >= 1).")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: observations must be finite.")
        if length is not None and value.shape[0] != length:
            raise ValueError("All modalities in a cell must have the same time points.")
        length = value.shape[0]
        result[name] = value
    return result


def split_pairs(cell_names, val_fraction=0.2, seed=42, n_cells=None,
                time_per_cell=None):
    """Split actual cell IDs; time_per_cell counts T-1 training pairs, not T."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")
    names = np.asarray(cell_names, dtype=object)
    cells = list(dict.fromkeys(cell_names))
    if len(cells) < 2:
        raise ValueError("Cell-level validation requires at least two cells.")
    if (n_cells is None) != (time_per_cell is None):
        raise ValueError("Provide both n_cells and time_per_cell, or neither.")
    if n_cells is not None:
        counts = [np.count_nonzero(names == c) for c in cells]
        if (n_cells != len(cells) or time_per_cell < 1
                or any(n != time_per_cell for n in counts)):
            raise ValueError("n_cells/time_per_cell do not match the dataset. "
                             "time_per_cell is the number of T-1 training pairs; "
                             "omit both arguments for unequal cell lengths.")
    n_val = min(len(cells) - 1, max(1, int(len(cells) * val_fraction)))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(cells), n_val, replace=False)
    val_cells = [cells[i] for i in selected]
    mask = np.isin(names, val_cells)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def integer_index(index):
    """Accept an integer or a one-element sequence for one latent coordinate."""
    values = np.asarray(index).reshape(-1)
    if values.size != 1 or values[0] != int(values[0]) or values[0] < 0:
        raise ValueError("index must identify one nonnegative latent coordinate.")
    return int(values[0])
