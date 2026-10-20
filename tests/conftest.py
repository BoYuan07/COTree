import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)

@pytest.fixture
def samples():
    rng = np.random.default_rng(42)
    out = []
    for i in range(8):
        arrays = [torch.tensor(rng.uniform(1, 4, (12, d)), dtype=torch.float32)
                  for d in (4, 3, 5, 4)]
        arrays[-1][:, -1] = 2.0  # invariant metabolite
        out.append((*arrays, f'cell{i}', 'group'))
    return out

@pytest.fixture
def frames(samples):
    return {s[-2]: {m: pd.DataFrame(t.numpy().T)
                    for m, t in zip(('genomics', 'transcriptomics', 'proteomics', 'metabolomics'), s[:4])}
            for s in samples}

@pytest.fixture
def four_omics(samples):
    from cotree import dtmae_flex as flex
    data = flex.training_dataset(samples, ['G', 'R', 'P', 'M'], {
        'G': {'scale_factor': 0.5}, 'M': {'filter_zero_var': True},
        'P': {'keep_indices': [0, 2, 4]},
    })
    dims = {m: x.numel() for m, x in data[0][0].items()}
    latent = dict(G=2, M=2, P=2)
    hidden = {m: [8] for m in 'GMPR'}
    torch.manual_seed(7)
    model = flex.autoencoder(dims, list('GMP'), ['R'], latent, hidden, hidden, hidden,
        dict(G=['G','M','P'], M=['M','P'], P=['R','M','P']), 0, normalize_input=['M','P'])
    model.eval()
    return data, dims, latent, hidden, model

@pytest.fixture
def branch_embeddings():
    t = np.linspace(0, 4, 18)
    samples = []
    for sign in (-1, 1):
        for i in range(4):
            y = sign * np.maximum(t - 1.4, 0) * (1 + i * 0.025)
            samples.append(np.column_stack((t, y, 0.12 * t ** 2)))
    return np.vstack(samples), len(samples), len(t)
