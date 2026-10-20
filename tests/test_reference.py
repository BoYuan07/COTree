"""Numerical parity checks against the unchanged local manuscript repository.

Set COTREE_MANUSCRIPT_DIR when running these tests outside the sibling checkout.
Only these comparison tests skip when the research source is unavailable.
"""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import cotree

SOURCE = Path(os.environ.get('COTREE_MANUSCRIPT_DIR', Path(__file__).resolve().parents[2] / 'github_manuscript_code_clean'))

def reference(name):
    path = SOURCE / f'{name}.py'
    if not path.exists():
        pytest.skip('Set COTREE_MANUSCRIPT_DIR to run manuscript comparisons')
    if name == 'DTMAE':
        pytest.importorskip('umap')
    spec = importlib.util.spec_from_file_location(f'reference_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_and_preprocessing_match_manuscript(samples, four_omics):
    ref = reference('DTMAE')
    data, dims, latent, hidden, model = four_omics
    original = ref.training_dataset(samples, [0, 2, 4])
    for (curr, nxt), expected in zip(data, original.samples):
        actual = [curr[m] for m in 'GMPR'] + [torch.cat([nxt[m] for m in 'GMP']),
                                             torch.cat([curr[m] for m in 'GMP'])]
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
    torch.manual_seed(7)
    original_model = ref.autoencoder(dims, hidden, hidden, hidden,
                                    {**latent, 'R':dims['R']}, dims, 0)
    original_model.eval()
    inputs = {m: torch.stack([data[i][0][m] for i in range(8)]) for m in 'GMPR'}
    embeddings, z, recon, pred = model(inputs)
    expected = original_model(*(inputs[m] for m in 'GMPR'))
    actual = [torch.cat([pred[m] for m in 'GMP'],dim=-1),
              torch.cat([recon[m] for m in 'GMP'],dim=-1),
              *[embeddings[m] for m in 'GMP'], z]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    x = torch.cat([inputs[m] for m in 'GMP'],dim=-1)
    torch.testing.assert_close(cotree.graph_manifold_loss(x, z), ref.graph_manifold_loss(x, z))


def test_tree_matches_manuscript(branch_embeddings):
    ref = reference('tree')
    X, C, T = branch_embeddings
    kwargs = dict(n_before=5, K_max=3, K_min=2, max_iter=1,
                  sample_num=20, epsilon=0.01, alpha_range=[0, 0, 1])
    expected = ref.construct_principal_tree(X, T, C, **kwargs)
    actual = cotree.construct_principal_tree(X, T, C, **kwargs)
    assert actual[0].keys() == expected[0].keys()
    for name in actual[0]:
        np.testing.assert_allclose(actual[0][name], expected[0][name], atol=1e-7)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-7)
    np.testing.assert_allclose(actual[4], expected[4], atol=1e-7)
    assert actual[3] == expected[3]


def test_driver_threshold_matches_manuscript():
    import importlib
    ref = reference('driver_species_detection')
    packaged = importlib.import_module('cotree.driver_species_detection')
    b = torch.tensor([4., 2., 1., 0.2])
    loss = lambda value: ((b - value)**2).sum()
    expected = ref._select_driver_threshold(b, loss, n_grid=8)
    actual = packaged._select_driver_threshold(b, loss, n_grid=8)
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1])
    np.testing.assert_allclose(actual[2]['R'], expected[2]['R'])
