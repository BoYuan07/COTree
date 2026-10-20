"""Dataset loading, manifold loss, and lineage decoding.

Model construction, preprocessing, training and embedding extraction live in
``cotree.dtmae_flex``.
"""
import torch
from torch.utils.data import Dataset


import pandas as pd
import numpy as np


from ._validation import validate_series, split_pairs
from ._model_api import decode_gmp


################################################################################DT-MAE

class OmicsDataset(Dataset):
    """Load four-omics cell trajectories from CSV folders or a data dictionary.

    CSV rows are species and columns are ordered time points. ``labels`` names
    cell-group subfolders. For in-memory data, it may instead map cell IDs to
    labels; otherwise cell IDs must start with a label (or use one common label).
    ``deathfilter`` implements the manuscript's dataset-specific metabolite-80
    truncation and therefore requires at least 81 metabolite features.
    """
    def __init__(self, data_folder, labels, deathfilter=False, data=None):
        self.data_folder, self.labels, self.death = data_folder, labels, deathfilter
        self.names, self.data, self.cell_label = {}, {}, []
        if data is None:
            self.load_datasets()
        else:
            self.data = {cell: dict(frames) for cell, frames in data.items()}
            for cell in self.data:
                if isinstance(labels, dict):
                    if cell not in labels:
                        raise ValueError(f"Missing label for cell {cell!r}.")
                    label = labels[cell]
                else:
                    matches = [v for v in labels if str(cell).startswith(str(v))]
                    if matches:
                        label = max(matches, key=lambda v: len(str(v)))
                    elif len(labels) == 1:
                        label = labels[0]
                    else:
                        raise ValueError("Pass a cell-ID-to-label mapping for in-memory data.")
                self.cell_label.append(label)
        if not self.data:
            raise ValueError("No cell trajectories found in the supplied data.")
        self.samples = self.create_pairs()

    def load_datasets(self):
        from pathlib import Path
        for label in self.labels:
            folder = Path(self.data_folder) / label
            for modality in ('genomics', 'transcriptomics', 'proteomics', 'metabolomics'):
                for file in sorted((folder / modality).glob('*.csv')):
                    cell_id = file.stem.rsplit('.', 1)[-1]
                    cell = str(label) + cell_id
                    if cell not in self.data:
                        self.data[cell] = {}
                        self.cell_label.append(label)
                    self.data[cell][modality] = pd.read_csv(file, index_col=0)

    def create_pairs(self):
        samples = []
        modalities = ('genomics', 'transcriptomics', 'proteomics', 'metabolomics')
        for (cell, frames), label in zip(self.data.items(), self.cell_label):
            if any(m not in frames for m in modalities):
                raise ValueError(f"Cell {cell!r} is missing an omics modality.")
            for m in modalities:
                if m in self.names and not self.names[m].equals(frames[m].index):
                    raise ValueError(f"Species order differs between cells for {m}.")
                self.names[m] = frames[m].index.copy()
            tensors = validate_series({m: frames[m].to_numpy().T for m in modalities}, modalities)
            if self.death:
                metabolome = tensors['metabolomics']
                if metabolome.shape[1] <= 80:
                    raise ValueError("deathfilter requires metabolite feature 80.")
                zeros = torch.nonzero(metabolome[:, 80] == 0).flatten()
                if zeros.numel():
                    end = int(zeros[0])
                    if end < 2:
                        raise ValueError(f"Cell {cell!r} has fewer than two pre-death observations.")
                    tensors = {m: t[:end] for m, t in tensors.items()}
                    for m in modalities:
                        self.data[cell][m] = frames[m].iloc[:, :end].copy()
            samples.append(tuple(tensors[m] for m in modalities) + (cell, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.cell_label[idx]


def cell_trajectory_val_split(n_cells, time_per_cell, val_fraction=0.2, seed=42):
    """Split equal-length cells; time_per_cell counts training pairs (T-1)."""
    if int(n_cells) != n_cells or int(time_per_cell) != time_per_cell or time_per_cell < 1:
        raise ValueError("n_cells and time_per_cell must be positive integers.")
    names = np.repeat(np.arange(n_cells), time_per_cell).tolist()
    train, val = split_pairs(names, val_fraction, seed)
    return train, val, np.unique(np.asarray(names)[train]), np.unique(np.asarray(names)[val])


def graph_manifold_loss(raw_inputs, latent_embeds, sigma=None, k=None, normalize=True):
    """
    Graph-based manifold regularization loss.

    Builds an input-space graph W and penalizes large latent distances
    between input-space neighbors.

    L = sum_ij W_ij ||z_i - z_j||^2
    """
    if raw_inputs.ndim != 2 or latent_embeds.ndim != 2 or len(raw_inputs) != len(latent_embeds):
        raise ValueError("Raw and latent arrays must be 2D with matching rows.")
    if len(raw_inputs) < 2:
        return latent_embeds.sum() * 0.0
    std = raw_inputs.std(dim=0, keepdim=True).clamp(min=1e-6)
    raw_inputs = raw_inputs / std

    raw_sq = torch.cdist(raw_inputs, raw_inputs).pow(2) / raw_inputs.shape[1]
    lat_sq = torch.cdist(latent_embeds, latent_embeds).pow(2) / latent_embeds.shape[1]

    n = raw_inputs.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=raw_inputs.device)
    off_diag = ~eye

    if sigma is None:
        flat = raw_sq[off_diag]
        sigma = flat.median().sqrt().detach().clamp(min=1e-6)

    W = torch.exp(-raw_sq / (2 * sigma ** 2))
    W = W.masked_fill(eye, 0.0)

    if k is not None:
        vals, idx = torch.topk(W, k=min(k, n - 1), dim=1)
        W_knn = torch.zeros_like(W)
        W_knn.scatter_(1, idx, vals)

        W = torch.maximum(W_knn, W_knn.T)

    W = W.detach()

    loss = (W * lat_sq).sum()

    if normalize:
        loss = loss / W.sum().clamp(min=1e-8)

    return loss


# Package convenience functions

def seperate_output(output, input_size):
    G = output[:,:input_size["G"]]
    M = output[:,input_size["G"]:(input_size["G"]+input_size["M"])]
    P = output[:, input_size['G'] + input_size['M']:input_size['G'] + input_size['M'] + input_size['P']]
    R = output[:, input_size["G"] + input_size["M"] + input_size["P"]:]
    return G, M, P, R

def reconstruct_lineage(trajectory_tree, model, device="cpu", interval=3):
    model.to(device).eval()
    resample_lineage_species = {}
    with torch.no_grad():
        for leaf, curve in trajectory_tree.items():
            embedding = torch.as_tensor(curve, dtype=torch.float32, device=device)
            reconstructed = decode_gmp(model, embedding)
            resample_lineage_species[leaf] = reconstructed.cpu().numpy()[::interval]
    return resample_lineage_species
