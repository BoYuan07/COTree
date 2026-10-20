from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from typing import Dict, List, Optional
from ._validation import validate_series, split_pairs
from .DTMAE import graph_manifold_loss


# ── building blocks ────────────────────────────────────────────────────────────

def relu_block(input_size: int, output_size: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LayerNorm(output_size),
        nn.LeakyReLU(0.05),
        nn.Dropout(dropout),
    )


def linear_block(input_size: int, output_size: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LayerNorm(output_size),
        nn.Dropout(dropout),
    )


class FCLayer(nn.Module):
    def __init__(self, layer_dims: List[int], dropout_rate: float, block_type: str = "relu"):
        super().__init__()
        self.net = nn.Sequential()
        builder = relu_block if block_type == "relu" else linear_block
        for i in range(len(layer_dims) - 1):
            self.net.add_module(f"block_{i}", builder(layer_dims[i], layer_dims[i + 1], dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    One FCLayer encoder per encoded modality, each mapping raw_dim → latent_dim.

    Parameters
    ----------
    raw_dims : dict[str, int]
        Raw feature dimension of each encoded modality.
    hidden_layers : dict[str, list[int]]
        Intermediate hidden widths per modality.  Missing keys → no hidden layers.
    latent_dims : dict[str, int]
        Latent embedding dimension per modality.
    dropout_rate : float
    normalize_input : list[str]
        Modality names whose raw input should be LayerNorm-normalised before
        encoding (useful for modalities with very different feature scales).
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        hidden_layers: Dict[str, List[int]],
        latent_dims: Dict[str, int],
        dropout_rate: float,
        normalize_input: Optional[List[str]] = None,
    ):
        super().__init__()
        self.modality_names = list(raw_dims.keys())
        normalize_input = normalize_input or []

        self.input_norms = nn.ModuleDict()
        self.encoders    = nn.ModuleDict()

        for m in self.modality_names:
            if m in normalize_input:
                self.input_norms[m] = nn.LayerNorm(raw_dims[m])
            dims = [raw_dims[m]] + hidden_layers.get(m, []) + [latent_dims[m]]
            self.encoders[m] = FCLayer(dims, dropout_rate)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Returns
        -------
        embeddings    : dict mod → tensor(batch, latent_dim)
        all_embedding : tensor(batch, sum_of_latent_dims)  — concatenation in modality order
        """
        embeddings: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            x = inputs[m]
            if m in self.input_norms:
                x = self.input_norms[m](x)
            embeddings[m] = self.encoders[m](x)
        all_embedding = torch.cat([embeddings[m] for m in self.modality_names], dim=-1)
        return embeddings, all_embedding


# ── decoder ────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    One FCLayer decoder per encoded modality, each mapping latent_dim → raw_dim.
    """

    def __init__(
        self,
        latent_dims: Dict[str, int],
        hidden_layers: Dict[str, List[int]],
        raw_dims: Dict[str, int],
        dropout_rate: float,
    ):
        super().__init__()
        self.modality_names = list(latent_dims.keys())
        self.decoders = nn.ModuleDict()
        for m in self.modality_names:
            dims = [latent_dims[m]] + hidden_layers.get(m, []) + [raw_dims[m]]
            self.decoders[m] = FCLayer(dims, dropout_rate)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns dict mod → tensor(batch, raw_dim)."""
        return {m: self.decoders[m](embeddings[m]) for m in self.modality_names}


# ── predictor ──────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Predicts each output modality's next-step values from a user-specified
    combination of latent embeddings and raw passthrough tensors.

    Parameters
    ----------
    predictor_inputs : dict[str, list[str]]
        For each output modality key, an ordered list of input names to
        concatenate.  Names present in ``latent_dims`` use their latent
        embedding; names present in ``passthrough_dims`` use the raw tensor.
    latent_dims : dict[str, int]
        Latent embedding dimension of each encoded modality.
    passthrough_dims : dict[str, int]
        Raw feature dimension of each passthrough modality.
    hidden_layers : dict[str, list[int]]
        Hidden widths per output modality head.
    output_dims : dict[str, int]
        Output (raw) dimension per predicted modality.
    dropout_rate : float

    Example
    -------
        predictor_inputs = {
            'G': ['G', 'M', 'P'],   # G latent + M latent + P latent
            'M': ['M', 'P'],
            'P': ['R', 'M', 'P'],   # raw R  + M latent  + P latent
        }
    """

    def __init__(
        self,
        predictor_inputs: Dict[str, List[str]],
        latent_dims: Dict[str, int],
        passthrough_dims: Dict[str, int],
        hidden_layers: Dict[str, List[int]],
        output_dims: Dict[str, int],
        dropout_rate: float,
    ):
        super().__init__()
        self.predictor_inputs = predictor_inputs
        self.latent_dims      = latent_dims
        self.passthrough_dims = passthrough_dims
        self.heads = nn.ModuleDict()

        for out_m, in_list in predictor_inputs.items():
            in_dim = sum(
                latent_dims[m] if m in latent_dims else passthrough_dims[m]
                for m in in_list
            )
            dims = [in_dim] + hidden_layers.get(out_m, []) + [output_dims[out_m]]
            self.heads[out_m] = FCLayer(dims, dropout_rate)

    def forward(
        self,
        embeddings:   Dict[str, torch.Tensor],
        passthroughs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        embeddings   : dict encoded_mod → tensor(batch, latent_dim)
        passthroughs : dict passthrough_mod → tensor(batch, raw_dim)
        Returns dict output_mod → tensor(batch, raw_dim)
        """
        all_sources = {**embeddings, **passthroughs}
        return {
            out_m: self.heads[out_m](
                torch.cat([all_sources[m] for m in in_list], dim=-1)
            )
            for out_m, in_list in self.predictor_inputs.items()
        }


# ── full model ─────────────────────────────────────────────────────────────────

class autoencoder(nn.Module):
    """
    DTMAE supporting any number of modalities with configurable predictor inputs.

    Parameters
    ----------
    raw_dims : dict[str, int]
        Raw feature dimension of every modality (encoded + passthrough).
    encoded_modalities : list[str]
        Modalities that pass through encoder → decoder (reconstruction path).
    passthrough_modalities : list[str]
        Modalities fed directly to the predictor without encoding.
    latent_dims : dict[str, int]
        Latent dimension per encoded modality.
    hidden_encoder : dict[str, list[int]]
    hidden_decoder : dict[str, list[int]]
    hidden_predictor : dict[str, list[int]]
    predictor_inputs : dict[str, list[str]]
        For each predicted output modality, an ordered list of input names
        (encoded → latent embedding used; passthrough → raw tensor used).
    dropout_rate : float
    normalize_input : list[str] or None
        Encoded modalities whose raw input gets a LayerNorm before the encoder.
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        encoded_modalities: List[str],
        passthrough_modalities: List[str],
        latent_dims: Dict[str, int],
        hidden_encoder: Dict[str, List[int]],
        hidden_decoder: Dict[str, List[int]],
        hidden_predictor: Dict[str, List[int]],
        predictor_inputs: Dict[str, List[str]],
        dropout_rate: float = 0.1,
        normalize_input: Optional[List[str]] = None,
    ):
        super().__init__()
        self.raw_dims = dict(raw_dims)
        self.latent_dims = {m: latent_dims[m] for m in encoded_modalities}
        self.encoded_modalities    = encoded_modalities
        self.passthrough_modalities = passthrough_modalities
        self.predictor_output_mods = list(predictor_inputs.keys())

        enc_raw    = {m: raw_dims[m]    for m in encoded_modalities}
        enc_latent = {m: latent_dims[m] for m in encoded_modalities}
        pass_raw   = {m: raw_dims[m]    for m in passthrough_modalities}
        pred_out   = {m: raw_dims[m]    for m in self.predictor_output_mods}

        self.encoder = Encoder(
            enc_raw, hidden_encoder, enc_latent,
            dropout_rate, normalize_input=normalize_input,
        )
        self.decoder = Decoder(
            enc_latent, hidden_decoder, enc_raw, dropout_rate,
        )
        self.predictor = Predictor(
            predictor_inputs, enc_latent, pass_raw,
            hidden_predictor, pred_out, dropout_rate,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Parameters
        ----------
        inputs : dict modality_name → tensor(batch, dim)
            Must contain all encoded and passthrough modalities.

        Returns
        -------
        embeddings    : dict encoded_mod → tensor(batch, latent_dim)
        all_embedding : tensor(batch, sum_latent_dims)
        reconstructed : dict encoded_mod → tensor(batch, raw_dim)
        predicted     : dict pred_mod → tensor(batch, raw_dim)
        """
        enc_in  = {m: inputs[m] for m in self.encoded_modalities}
        pass_in = {m: inputs[m] for m in self.passthrough_modalities}

        embeddings, all_embedding = self.encoder(enc_in)
        reconstructed = self.decoder(embeddings)
        predicted     = self.predictor(embeddings, pass_in)

        return embeddings, all_embedding, reconstructed, predicted


# ── dataset ────────────────────────────────────────────────────────────────────

class training_dataset(Dataset):
    """
    Converts a list of per-cell multi-omics time series into (current_t, next_t) sample pairs.

    Parameters
    ----------
    samples : list
        Either a list of dicts with modality-name keys and tensor values of
        shape (T, D), plus optional 'cell_id' and 'label' entries; or a list
        of tuples ``(*modality_tensors, cell_id, label)`` whose positional
        order matches ``modality_names``.
    modality_names : list[str]
        Ordered modality names.  For tuple-format samples, defines position → name.
    modality_config : dict[str, dict] or None
        Per-modality preprocessing.  Recognised keys per modality:

        ``'normalize'``       bool (default True) — divide by mean value
        ``'scale_factor'``    float (default 1.0) — extra divisor applied on
                              top of the mean, e.g. 0.5 replicates the original
                              genomics scaling
        ``'filter_zero_var'`` bool (default False) — drop features with zero
                              variance across the full dataset
        ``'keep_indices'``    list[int] or None — subset of feature columns to
                              retain before other preprocessing steps

    Attributes
    ----------
    scale_factors : dict[str, torch.Tensor]
        Scalar used to normalise each modality (mean * scale_factor).
    active_mask   : dict[str, torch.Tensor or None]
        Boolean mask or integer index list actually applied to each modality.
    feature_indices : dict[str, torch.Tensor]
        Retained indices in the original feature order, for aligning species names.
    raw_dims : dict[str, int]
        Feature widths before preprocessing, checked for every input trajectory.
    cell_names    : list[str]
        Cell identifier for every flat time-step sample (length = n_pairs).
    """

    def __init__(
        self,
        samples: list,
        modality_names: List[str],
        modality_config: Optional[Dict[str, dict]] = None,
    ):
        if not modality_names or len(set(modality_names)) != len(modality_names):
            raise ValueError("modality_names must be nonempty and unique.")
        self.modality_names  = modality_names
        self.modality_config = modality_config or {}
        self.scale_factors: Dict[str, torch.Tensor] = {}
        self.active_mask:   Dict[str, Optional[torch.Tensor]] = {}
        self.feature_indices: Dict[str, torch.Tensor] = {}

        parsed = self._parse_samples(samples)
        self.raw_dims = {m: parsed[0][m].shape[1] for m in self.modality_names}
        self._items, self.cell_names = self._build_pairs(parsed)

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_samples(self, samples: list) -> List[dict]:
        if len(samples) == 0:
            raise ValueError("At least one cell trajectory is required.")
        result = []
        for sample_index, s in enumerate(samples):
            if isinstance(s, dict):
                d = dict(s)
                d.update(validate_series(d, self.modality_names))
                d.setdefault('cell_id', f'cell_{sample_index}')
                result.append(d)
            else:
                n_mods = len(self.modality_names)
                if len(s) < n_mods:
                    raise ValueError("Each sample must contain all requested modalities.")
                d = {self.modality_names[i]: s[i] for i in range(n_mods)}
                if len(s) > n_mods:
                    d['cell_id'] = s[n_mods]
                if len(s) > n_mods + 1:
                    d['label'] = s[n_mods + 1]
                d.update(validate_series(d, self.modality_names))
                d.setdefault('cell_id', f'cell_{sample_index}')
                result.append(d)
        expected_dims = getattr(self, 'raw_dims', None) or {
            m: result[0][m].shape[1] for m in self.modality_names
        }
        for d in result:
            self._check_feature_dimensions(d, expected_dims)
        ids = [d['cell_id'] for d in result]
        if len(set(ids)) != len(ids):
            raise ValueError("Each input trajectory must have a distinct cell_id.")
        return result

    @staticmethod
    def _check_feature_dimensions(series, expected_dims):
        for m, width in expected_dims.items():
            if series[m].shape[1] != width:
                raise ValueError(
                    f"{m}: expected {width} raw features, got {series[m].shape[1]}. "
                    "Use the same feature count and order as the training data."
                )

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _build_pairs(self, parsed: List[dict]):
        # Separate current (t) and next (t+1) for every cell
        curr_list: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modality_names}
        next_list: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modality_names}
        cell_names = []

        for d in parsed:
            cell_id = d.get('cell_id', f'cell_{len(cell_names)}')
            T = d[self.modality_names[0]].shape[0]
            for m in self.modality_names:
                t = d[m]
                curr_list[m].append(t[:-1])
                next_list[m].append(t[1:])
            cell_names.extend([cell_id] * (T - 1))

        curr = {m: torch.cat(curr_list[m], dim=0) for m in self.modality_names}
        nxt  = {m: torch.cat(next_list[m], dim=0) for m in self.modality_names}

        # Per-modality preprocessing
        for m in self.modality_names:
            cfg = self.modality_config.get(m, {})
            retained = torch.arange(curr[m].shape[1])

            # 1. optional column subset
            ki = cfg.get('keep_indices', None)
            if ki is not None:
                idx = torch.as_tensor(ki)
                if idx.ndim != 1 or not idx.numel() or idx.dtype == torch.bool or (idx != idx.long()).any() or (idx < 0).any() or (idx >= curr[m].shape[1]).any():
                    raise ValueError(f'Invalid keep_indices for {m}.')
                idx = idx.long()
                retained = retained[idx]
                curr[m] = curr[m][:, idx]
                nxt[m]  = nxt[m][:, idx]
                self.active_mask[m] = idx
            else:
                self.active_mask[m] = None

            # 2. optional zero-variance filter (applied after keep_indices)
            if cfg.get('filter_zero_var', False):
                var  = torch.var(torch.cat([curr[m], nxt[m]], dim=0), dim=0)
                mask = var > 0
                retained = retained[mask]
                curr[m] = curr[m][:, mask]
                nxt[m]  = nxt[m][:, mask]
                # merge with any existing active_mask
                prev = self.active_mask[m]
                if prev is not None:
                    # map bool mask back into the already-subsetted index space
                    self.active_mask[m] = (prev, mask)   # stored as tuple for preprocess_sample
                else:
                    self.active_mask[m] = mask

            if nxt[m].shape[1] == 0:
                raise ValueError(f"Preprocessing removed all features of {m}.")
            self.feature_indices[m] = retained

            # 3. mean-scaling
            if cfg.get('normalize', True):
                scale_factor = cfg.get('scale_factor', 1.0)
                if not np.isfinite(scale_factor) or scale_factor <= 0:
                    raise ValueError('scale_factor must be positive and finite.')
                mean_val = torch.mean(nxt[m]).clamp(min=1e-8)
                self.scale_factors[m] = mean_val * scale_factor
            else:
                self.scale_factors[m] = torch.tensor(1.0)

            curr[m] = curr[m] / self.scale_factors[m]
            nxt[m]  = nxt[m]  / self.scale_factors[m]

        # Flatten into list of (curr_dict, next_dict) tuples
        n = curr[self.modality_names[0]].shape[0]
        items = [
            ({m: curr[m][i] for m in self.modality_names},
             {m: nxt[m][i]  for m in self.modality_names})
            for i in range(n)
        ]
        return items, cell_names

    # ── preprocessing of new samples using fitted scalers ─────────────────────

    def preprocess_sample(self, sample_dict: dict) -> Dict[str, torch.Tensor]:
        """
        Apply stored masks and scale factors to a single-cell time series dict.
        Used by ``extract_latent_embeddings`` to ensure consistent preprocessing.

        Parameters
        ----------
        sample_dict : dict mod → tensor(T, D)

        Returns
        -------
        dict mod → tensor(T, D_proc)
        """
        sample_dict = validate_series(sample_dict, self.modality_names)
        self._check_feature_dimensions(sample_dict, self.raw_dims)
        result: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            t = sample_dict[m]
            mask = self.active_mask.get(m)

            if mask is None:
                pass
            elif isinstance(mask, tuple):
                # keep_indices + zero_var mask applied sequentially
                idx_mask, bool_mask = mask
                t = t[:, idx_mask]
                t = t[:, bool_mask]
            elif mask.dtype == torch.bool:
                t = t[:, mask]
            else:
                t = t[:, mask]      # integer index tensor

            t = t / self.scale_factors[m]
            result[m] = t
        return result

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]

    def transform_samples(self, raw_samples: list) -> Dict[str, torch.Tensor]:
        """Stack all time points using the fitted feature masks and scales.

        Returns modality tensors in cell order, including the final observation
        of each cell. These are the processed inputs used by embedding extraction
        and by functional-module/species analyses, not unscaled raw counts.
        """
        cells = [self.preprocess_sample(d) for d in self._parse_samples(raw_samples)]
        return {m: torch.cat([d[m] for d in cells], dim=0) for m in self.modality_names}


# ── collation helper ───────────────────────────────────────────────────────────

def collate_omics(batch):
    """
    DataLoader collate_fn for training_dataset.
    Returns (curr_batch, next_batch), each a dict mod → tensor(B, D).
    """
    curr_batch: Dict[str, List[torch.Tensor]] = {}
    next_batch: Dict[str, List[torch.Tensor]] = {}
    for curr_d, next_d in batch:
        for m, v in curr_d.items():
            curr_batch.setdefault(m, []).append(v)
        for m, v in next_d.items():
            next_batch.setdefault(m, []).append(v)
    curr_out = {m: torch.stack(vs) for m, vs in curr_batch.items()}
    next_out = {m: torch.stack(vs) for m, vs in next_batch.items()}
    return curr_out, next_out


# ── training loop ──────────────────────────────────────────────────────────────

def _compute_loss(
    model: autoencoder,
    curr_batch: Dict[str, torch.Tensor],
    next_batch: Dict[str, torch.Tensor],
    lambda_manifold: float = 0.0,
    manifold_raw_key: Optional[str] = None,
    gamma: Optional[Dict[str, float]] = None,
    eta_curr: float = 0.5,
    return_parts: bool = False,
):
    """Single forward + loss computation shared by train and eval steps.

    Matches the weighted objective used in ``main_revised``:

        loss = eta_curr * sum_m gamma_m * mse(recon_m, curr_m)
             + (1 - eta_curr) * sum_m gamma_m * mse(pred_m, next_m)

    ``gamma`` is a per-modality weight dict (missing modalities default to 1.0);
    ``eta_curr`` balances reconstruction (current) vs prediction (next step).
    With the defaults (gamma=None -> all 1.0, eta_curr=0.5) the loss reduces to
    ``0.5 * (recon + pred)``.

    With ``return_parts=True`` also returns a dict of unweighted per-modality
    MSE values (floats), keyed ``recon_<m>`` and ``pred_<m>``, plus the
    aggregate ``recon``, ``pred`` and ``manifold`` terms.
    """
    embeddings, all_embedding, reconstructed, predicted = model(curr_batch)

    def gw(m: str) -> float:
        return 1.0 if gamma is None else float(gamma.get(m, 1.0))

    recon_terms = {
        m: nn.functional.mse_loss(reconstructed[m], curr_batch[m])
        for m in model.encoded_modalities
    }
    pred_terms = {
        m: nn.functional.mse_loss(predicted[m], next_batch[m])
        for m in model.predictor_output_mods
    }

    recon_loss = sum(gw(m) * t for m, t in recon_terms.items())
    pred_loss  = sum(gw(m) * t for m, t in pred_terms.items())
    loss = eta_curr * recon_loss + (1.0 - eta_curr) * pred_loss

    man_loss = None
    if lambda_manifold > 0.0:
        raw_key = manifold_raw_key or model.encoded_modalities[0]
        man_loss = graph_manifold_loss(curr_batch[raw_key], all_embedding)
        loss = loss + lambda_manifold * man_loss

    if not return_parts:
        return loss

    parts = {f'recon_{m}': float(t.item()) for m, t in recon_terms.items()}
    parts.update({f'pred_{m}': float(t.item()) for m, t in pred_terms.items()})
    parts['recon'] = float(recon_loss.item())
    parts['pred']  = float(pred_loss.item())
    parts['manifold'] = 0.0 if man_loss is None else float(man_loss.item())
    return loss, parts


def train_dtmae(
    model: autoencoder,
    dataset: training_dataset,
    n_cells: Optional[int] = None,
    time_per_cell: Optional[int] = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_manifold: float = 0.0,
    manifold_raw_key: Optional[str] = None,
    gamma: Optional[Dict[str, float]] = None,
    eta_curr: float = 0.5,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    *,
    step_size: Optional[int] = None,
    scheduler_gamma: float = 0.5,
) -> tuple:
    """
    Train an autoencoder on a training_dataset.

    Train/validation always split by actual cell IDs, keeping trajectories
    together. Optional n_cells/time_per_cell validate equal-length inputs;
    time_per_cell counts T-1 pairs. Omit both for unequal trajectory lengths.

    Parameters
    ----------
    model   : autoencoder
    dataset : training_dataset
    n_cells          : int or None — number of cells (enables cell-level split)
    time_per_cell    : int or None — training pairs per cell (T-1; validates equal lengths)
    n_epochs         : int
    batch_size       : int
    lr               : float — Adam learning rate
    weight_decay     : float — Adam weight decay
    lambda_manifold  : float — weight for graph manifold regularisation (0 = off)
    manifold_raw_key : str or None — modality whose raw features define the
                       input-space graph; defaults to first encoded modality
    val_fraction     : float
    seed             : int
    device           : str
    verbose          : bool
    step_size        : optional epoch interval for StepLR; None disables scheduling
    scheduler_gamma  : StepLR decay factor, distinct from per-modality gamma weights

    Returns
    -------
    model   : trained autoencoder (on CPU)
    history : dict with 'train_loss' and 'val_loss' lists
    """
    device = torch.device(device)
    model.to(device)

    # ── train / val split ─────────────────────────────────────────────────────
    if n_epochs < 1 or batch_size < 1 or not 0 <= eta_curr <= 1:
        raise ValueError("n_epochs/batch_size must be positive and eta_curr must be in [0, 1].")
    if step_size is not None and (isinstance(step_size, bool) or
                                 not isinstance(step_size, (int, np.integer)) or step_size < 1):
        raise ValueError("step_size must be a positive integer or None.")
    if not np.isfinite(scheduler_gamma) or scheduler_gamma <= 0:
        raise ValueError("scheduler_gamma must be positive and finite.")
    torch.manual_seed(seed)
    train_idx, val_idx = split_pairs(dataset.cell_names, val_fraction, seed,
                                      n_cells, time_per_cell)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size,
        shuffle=True, collate_fn=collate_omics,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size,
        shuffle=False, collate_fn=collate_omics,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimiser, step_size, gamma=scheduler_gamma)
                 if step_size is not None else None)
    history = {'train_loss': [], 'val_loss': [], 'train_parts': [], 'val_parts': [], 'lr': []}

    def _run_epoch(loader, n_samples, training: bool):
        """Return (mean loss, dict of sample-weighted mean loss components)."""
        total = 0.0
        part_totals: Dict[str, float] = {}
        for curr_batch, next_batch in loader:
            curr_batch = {m: v.to(device) for m, v in curr_batch.items()}
            next_batch = {m: v.to(device) for m, v in next_batch.items()}
            if training:
                optimiser.zero_grad()
                loss, parts = _compute_loss(model, curr_batch, next_batch, lambda_manifold,
                                            manifold_raw_key, gamma, eta_curr,
                                            return_parts=True)
                loss.backward()
                optimiser.step()
            else:
                with torch.no_grad():
                    loss, parts = _compute_loss(model, curr_batch, next_batch, 0.0, None,
                                                gamma, eta_curr, return_parts=True)
            bs = curr_batch[model.encoded_modalities[0]].shape[0]
            total += loss.item() * bs
            for k, v in parts.items():
                part_totals[k] = part_totals.get(k, 0.0) + v * bs
        return total / n_samples, {k: v / n_samples for k, v in part_totals.items()}

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, train_parts = _run_epoch(train_loader, len(train_idx), training=True)

        model.eval()
        val_loss, val_parts = _run_epoch(val_loader, len(val_idx), training=False)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_parts'].append(train_parts)
        history['val_parts'].append(val_parts)
        history['lr'].append(optimiser.param_groups[0]['lr'])
        if scheduler is not None:
            scheduler.step()

        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == 1):
            pred_str = "  ".join(
                f"{m}:{train_parts[f'pred_{m}']:.4f}/{val_parts[f'pred_{m}']:.4f}"
                for m in model.predictor_output_mods
            )
            print(f"Epoch {epoch:4d}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  | pred (tr/val) {pred_str}")

    model.to(torch.device('cpu'))
    return model, history


def evaluate_dtmae(
    model: autoencoder,
    dataset: training_dataset,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 256,
    gamma: Optional[Dict[str, float]] = None,
    eta_curr: float = 0.5,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Per-modality loss breakdown for an already-trained model, without training.

    Parameters
    ----------
    model      : trained autoencoder
    dataset    : training_dataset
    indices    : array of sample indices to evaluate (None = whole dataset)
    batch_size : int
    gamma, eta_curr : same weighting as ``train_dtmae`` (affects only the
                 aggregate ``loss``/``recon``/``pred`` entries, not the raw
                 per-modality MSEs)
    device     : str

    Returns
    -------
    dict with keys ``loss``, ``recon``, ``pred``, ``manifold`` and one
    ``recon_<m>`` / ``pred_<m>`` entry per modality — all sample-weighted means.
    """
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(dev)

    data = dataset if indices is None else Subset(dataset, indices)
    if len(data) == 0:
        raise ValueError("Evaluation requires at least one observation.")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_omics)

    totals: Dict[str, float] = {}
    n_seen = 0
    with torch.no_grad():
        for curr_batch, next_batch in loader:
            curr_batch = {m: v.to(dev) for m, v in curr_batch.items()}
            next_batch = {m: v.to(dev) for m, v in next_batch.items()}
            loss, parts = _compute_loss(model, curr_batch, next_batch, 0.0, None,
                                        gamma, eta_curr, return_parts=True)
            bs = curr_batch[model.encoded_modalities[0]].shape[0]
            n_seen += bs
            parts['loss'] = float(loss.item())
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v * bs

    model.to(torch.device('cpu'))
    if was_training:
        model.train()
    return {k: v / n_seen for k, v in totals.items()}


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_latent_embeddings(
    model: autoencoder,
    raw_samples: list,
    dataset: training_dataset,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract the concatenated latent embedding for every time step of every cell.

    The same preprocessing (feature masking + scaling) as used during training
    is applied via ``dataset.preprocess_sample``.

    Parameters
    ----------
    model       : trained autoencoder
    raw_samples : list of dicts or tuples (same format as training_dataset input)
                  Each element is one cell's full time series.
    dataset     : fitted training_dataset (supplies scale factors and masks)
    batch_size  : int — time steps processed at once
    device      : str

    Returns
    -------
    flat_embeddings : np.ndarray of shape (n_cells * T, sum_latent_dims)
        Ready to pass directly to ``construct_principal_tree``. Keep unmodified
        model embeddings for functional-module and driver-species analyses.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    parsed = dataset._parse_samples(raw_samples)
    all_embeds = []

    with torch.no_grad():
        for d in parsed:
            proc = dataset.preprocess_sample(d)
            T = proc[dataset.modality_names[0]].shape[0]
            cell_parts = []
            for t0 in range(0, T, batch_size):
                t1 = min(t0 + batch_size, T)
                inputs = {m: proc[m][t0:t1].to(dev) for m in dataset.modality_names}
                _, all_embedding, _, _ = model(inputs)
                cell_parts.append(all_embedding.cpu().numpy())
            all_embeds.append(np.vstack(cell_parts))

    model.to(torch.device('cpu'))
    return np.vstack(all_embeds)


# ── visualisation ──────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train loss', linewidth=2)
    ax.plot(epochs, history['val_loss'],   label='Val loss',   linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss',  fontsize=12)
    ax.set_title('DTMAE training history', fontsize=13)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def vis_pca(
    flat_embeddings: np.ndarray,
    labels,
    pc_x: int = 0,
    pc_y: int = 1,
    alpha: float = 0.5,
    s: float = 0.8,
    save_path: Optional[str] = None,
):
    """PCA scatter of flat latent embeddings coloured by label."""
    pca = PCA(n_components=max(pc_x, pc_y) + 1)
    Z   = pca.fit_transform(flat_embeddings)

    labels_arr = np.asarray(labels)
    uniques    = list(OrderedDict.fromkeys(labels_arr.tolist()))
    palette    = ["#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
                  "#FFD92F", "#E5C494", "#B3B3B3"]
    color_map  = {cat: palette[i % len(palette)] for i, cat in enumerate(uniques)}

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5))
    for cat in uniques:
        m = labels_arr == cat
        ax.scatter(Z[m, pc_x], Z[m, pc_y],
                   c=color_map[cat], alpha=alpha, s=s, label=str(cat))
    ax.set_xlabel(f"PC {pc_x + 1}", fontsize=12)
    ax.set_ylabel(f"PC {pc_y + 1}", fontsize=12)
    ax.set_title("DTMAE latent space (PCA)", fontsize=13)
    ax.legend(markerscale=6, fontsize=9, frameon=True)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# ── hyperparameter tuning (optional — requires optuna) ─────────────────────────

def tune_dtmae(
    dataset: training_dataset,
    raw_dims: Dict[str, int],
    encoded_modalities: List[str],
    passthrough_modalities: List[str],
    predictor_inputs: Dict[str, List[str]],
    normalize_input: Optional[List[str]] = None,
    n_cells: Optional[int] = None,
    time_per_cell: Optional[int] = None,
    n_trials: int = 30,
    n_epochs_per_trial: int = 50,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    search_space: Optional[dict] = None,
    direction: str = "minimize",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    show_progress_bar: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Hyperparameter search for DTMAE using Optuna (optional dependency).

    Requires ``pip install optuna``.  If Optuna is not installed the function
    raises ``ImportError`` with an installation hint.

    The function builds a fresh :class:`autoencoder` for each trial, trains it
    for ``n_epochs_per_trial`` epochs, and evaluates validation prediction error,
    validation reconstruction error, and manifold loss.
    After all trials the best hyperparameters are returned as a dict that can
    be passed directly to ``autoencoder`` and ``train_dtmae``.

    Parameters
    ----------
    dataset : training_dataset
        Fitted dataset (preprocessing already applied).
    raw_dims : dict[str, int]
        Raw (post-preprocessing) feature dimensions — same as passed to
        :class:`autoencoder`.
    encoded_modalities : list[str]
    passthrough_modalities : list[str]
    predictor_inputs : dict[str, list[str]]
    normalize_input : list[str] or None
    n_cells : int or None
        Enables cell-level train/val split (recommended).
    time_per_cell : int or None
    n_trials : int
        Number of Optuna trials.
    n_epochs_per_trial : int
        Training epochs per trial (keep lower than final training for speed).
    val_fraction : float
    seed : int
    device : str
    search_space : dict or None
        Override default search ranges.  Supported keys and their defaults
        (mirrors ``main_revised``'s NSGA-II objective)::

            {
                # per-modality raw gamma weight (log-uniform), normalised to
                # sum to 1 across encoded modalities
                'gamma_low':  0.1,  'gamma_high':  1.0,

                # reconstruction/prediction balance eta_curr (log-uniform)
                'eta_low':  0.1,  'eta_high':  1.0,

                # single hidden-layer width per modality (categorical)
                'hidden_choices':  [128, 256],

                # latent dim for each encoded modality (categorical list)
                # key pattern: 'latent_<modality>'  e.g. 'latent_G'
                'latent_G':  [8, 16, 32, 64],

                # fixed (NOT tuned) — matches main_revised
                'lr':  0.005,  'batch_size':  200,  'dropout':  0.0,

                # NSGA-II population size
                'population_size':  20,
            }

        Any key absent from ``search_space`` falls back to the default above.
    direction : str
        Ignored — the search is multi-objective (all three objectives are
        minimised).  Kept for signature compatibility.
    study_name : str or None
        Optuna study name (useful for resuming).
    storage : str or None
        Optuna storage URL, e.g. ``"sqlite:///dtmae_study.db"`` to persist
        trials across Python sessions.
    show_progress_bar : bool
        Pass to ``optuna.Study.optimize``.
    verbose : bool
        Print best val loss after every trial.

    Returns
    -------
    best_params : dict
        The Pareto point minimising ``val_pred + val_curr``, as a ready-to-use
        dict with keys: ``latent_dims``, ``hidden_encoder``, ``hidden_decoder``,
        ``hidden_predictor``, ``gamma``, ``eta_curr``, ``lr``, ``batch_size``,
        ``dropout_rate``, ``lambda_manifold``.  Also attaches ``_study`` and
        ``_pareto_trials`` for inspecting the full Pareto front.

    Example
    -------
    ::

        best = tune_dtmae(
            dataset, raw_dims,
            encoded_modalities=['G', 'M', 'P'],
            passthrough_modalities=['R'],
            predictor_inputs={'G': ['G','M','P'], 'M': ['M','P'], 'P': ['R','M','P']},
            n_cells=cell_num, time_per_cell=time_length - 1,
            n_trials=30, n_epochs_per_trial=50,
            storage='sqlite:///dtmae_study.db',
        )

        model = autoencoder(
            raw_dims=raw_dims,
            encoded_modalities=['G', 'M', 'P'],
            passthrough_modalities=['R'],
            predictor_inputs={'G': ['G','M','P'], 'M': ['M','P'], 'P': ['R','M','P']},
            latent_dims=best['latent_dims'],
            hidden_encoder=best['hidden_encoder'],
            hidden_decoder=best['hidden_decoder'],
            hidden_predictor=best['hidden_predictor'],
            dropout_rate=best['dropout_rate'],
        )
        model, history = train_dtmae(
            model, dataset,
            n_cells=cell_num, time_per_cell=time_length - 1,
            lr=best['lr'], batch_size=best['batch_size'],
            gamma=best['gamma'], eta_curr=best['eta_curr'],
            lambda_manifold=best['lambda_manifold'],
        )
    """
    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )
    except ImportError:
        raise ImportError(
            "Optuna is required for tune_dtmae.  Install it with:\n"
            "    pip install cotree[tuning]"
        )

    sp = {
        'gamma_low': 0.1, 'gamma_high': 1.0,
        'eta_low': 0.1, 'eta_high': 1.0,
        'hidden_choices': [128, 256],
        'lr': 0.005,
        'batch_size': 200,
        'dropout': 0.0,
        'population_size': 20,
    }
    # per-modality latent dim defaults
    for m in encoded_modalities:
        sp[f'latent_{m}'] = [8, 16, 32, 64]
    if search_space:
        sp.update(search_space)

    predictor_output_mods = list(predictor_inputs.keys())

    # ── train / val split shared by all trials (for the val objectives) ───────
    _, val_idx = split_pairs(dataset.cell_names, val_fraction, seed,
                              n_cells, time_per_cell)

    def _three_objectives(model):
        """Reproduce main_revised's three Pareto objectives:
        (val prediction MSE, val reconstruction MSE, manifold loss on all data).
        Both validation losses are UNWEIGHTED sums over modalities."""
        dev = torch.device(device)
        model.eval(); model.to(dev)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=sp['batch_size'],
                                shuffle=False, collate_fn=collate_omics)
        all_loader = DataLoader(dataset, batch_size=sp['batch_size'],
                                shuffle=False, collate_fn=collate_omics)
        vp = vc = 0.0; n_tot = 0
        raw_parts, lat_parts = [], []
        with torch.no_grad():
            for curr, nxt in val_loader:
                curr = {m: v.to(dev) for m, v in curr.items()}
                nxt = {m: v.to(dev) for m, v in nxt.items()}
                _, _, recon, pred = model(curr)
                lc = sum(nn.functional.mse_loss(recon[m], curr[m])
                         for m in model.encoded_modalities)
                lp = sum(nn.functional.mse_loss(pred[m], nxt[m])
                         for m in model.predictor_output_mods)
                n = curr[model.encoded_modalities[0]].shape[0]
                vc += lc.item() * n; vp += lp.item() * n; n_tot += n
            for curr, nxt in all_loader:
                curr = {m: v.to(dev) for m, v in curr.items()}
                _, all_emb, _, _ = model(curr)
                raw = torch.cat([curr[m] for m in model.encoded_modalities], dim=-1)
                raw_parts.append(raw.cpu()); lat_parts.append(all_emb.cpu())
        man = graph_manifold_loss(torch.cat(raw_parts, 0), torch.cat(lat_parts, 0)).item()
        model.to(torch.device('cpu'))
        return vp / n_tot, vc / n_tot, man

    # ── objective (3 Pareto objectives, identical to main_revised) ────────────
    def objective(trial: "optuna.Trial"):
        # -- per-modality gamma weights (log-uniform raw, normalised to sum 1) --
        gamma_raw = {
            m: trial.suggest_float(f'gamma_{m}_raw', sp['gamma_low'], sp['gamma_high'], log=True)
            for m in encoded_modalities
        }
        gsum = sum(gamma_raw.values())
        gamma = {m: gamma_raw[m] / gsum for m in encoded_modalities}

        eta_curr = trial.suggest_float('eta_curr', sp['eta_low'], sp['eta_high'], log=True)

        # -- single hidden-layer width per modality --
        hidden = {m: trial.suggest_categorical(f'hidden_{m}', sp['hidden_choices'])
                  for m in encoded_modalities}
        hidden_encoder   = {m: [hidden[m]] for m in encoded_modalities}
        hidden_decoder   = {m: [hidden[m]] for m in encoded_modalities}
        hidden_predictor = {m: [hidden[m] if m in hidden else trial.suggest_categorical(f'hidden_{m}', sp['hidden_choices'])] for m in predictor_output_mods}

        # -- latent dims per modality --
        latent_dims: Dict[str, int] = {
            m: trial.suggest_categorical(f'latent_{m}', sp[f'latent_{m}'])
            for m in encoded_modalities
        }

        # -- build model --
        torch.manual_seed(seed)
        try:
            model = autoencoder(
                raw_dims               = raw_dims,
                encoded_modalities     = encoded_modalities,
                passthrough_modalities = passthrough_modalities,
                latent_dims            = latent_dims,
                hidden_encoder         = hidden_encoder,
                hidden_decoder         = hidden_decoder,
                hidden_predictor       = hidden_predictor,
                predictor_inputs       = predictor_inputs,
                dropout_rate           = sp['dropout'],
                normalize_input        = normalize_input,
            )
        except Exception as exc:
            raise optuna.exceptions.TrialPruned(f"Model build failed: {exc}")

        # -- train with the weighted (gamma/eta) loss --
        torch.manual_seed(seed)
        model, _ = train_dtmae(
            model, dataset,
            n_cells          = n_cells,
            time_per_cell    = time_per_cell,
            n_epochs         = n_epochs_per_trial,
            batch_size       = sp['batch_size'],
            lr               = sp['lr'],
            lambda_manifold  = 0.0,
            gamma            = gamma,
            eta_curr         = eta_curr,
            val_fraction     = val_fraction,
            seed             = seed,
            device           = device,
            verbose          = False,
        )

        # -- three Pareto objectives: val_pred, val_curr, manifold --
        return _three_objectives(model)

    # ── create / load multi-objective study and optimise ──────────────────────
    sampler = optuna.samplers.NSGAIISampler(population_size=sp['population_size'], seed=seed)
    study = optuna.create_study(
        directions = ['minimize', 'minimize', 'minimize'],   # val_pred, val_curr, manifold
        sampler    = sampler,
        study_name = study_name,
        storage    = storage,
        load_if_exists = storage is not None,
    )
    study.optimize(
        objective,
        n_trials          = n_trials,
        show_progress_bar = show_progress_bar,
    )

    # ── pick the best Pareto point by (val_pred + val_curr)─
    pareto = study.best_trials
    best_trial = min(pareto, key=lambda t: t.values[0] + t.values[1])
    bp = best_trial.params
    if verbose:
        print(f"\n{len(pareto)} Pareto-optimal trials; selected trial #{best_trial.number}")
        print(f"  val_pred={best_trial.values[0]:.6f}  val_curr={best_trial.values[1]:.6f}  "
              f"manifold={best_trial.values[2]:.6f}")
        print(f"  {bp}")

    g_raw = {m: bp[f'gamma_{m}_raw'] for m in encoded_modalities}
    g_sum = sum(g_raw.values())
    best_gamma  = {m: g_raw[m] / g_sum for m in encoded_modalities}
    best_latent = {m: bp[f'latent_{m}'] for m in encoded_modalities}
    best_hidden = {m: [bp[f'hidden_{m}']] for m in encoded_modalities}
    best_params = {
        'latent_dims':      best_latent,
        'hidden_encoder':   best_hidden,
        'hidden_decoder':   best_hidden,
        'hidden_predictor': {m: [bp[f'hidden_{m}']] for m in predictor_output_mods},
        'gamma':            best_gamma,
        'eta_curr':         bp['eta_curr'],
        'lr':               sp['lr'],
        'batch_size':       sp['batch_size'],
        'dropout_rate':     sp['dropout'],
        'lambda_manifold':  0.0,
        '_study':           study,          # attach study for further inspection
        '_pareto_trials':   pareto,
    }
    return best_params
