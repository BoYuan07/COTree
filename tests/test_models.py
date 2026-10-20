import numpy as np
import pytest
import torch
import cotree
from cotree import dtmae_flex as flex
from cotree._validation import split_pairs


def test_preprocessing_and_in_memory_loader(frames, samples, four_omics):
    ds = cotree.OmicsDataset(None, ['group'], data=frames)
    assert len(ds.samples) == len(samples)
    assert ds.cell_label == ['group'] * len(samples)
    td, dims, latent, hidden, model = four_omics
    assert len(td) == 8 * 11
    assert len(td.cell_names) == len(td)
    assert td.cell_names[:11] == ['cell0'] * 11
    assert dims == dict(G=4, M=3, P=3, R=3)
    processed = td.transform_samples(ds.samples)
    output = flex.extract_latent_embeddings(model, ds.samples, td)
    assert output.shape == (8 * 12, 6)
    assert np.isfinite(processed['M']).all()
    curves = cotree.reconstruct_lineage({'Lineage1': output[:12]}, model)
    assert curves['Lineage1'].shape == (4, 10)
    np.testing.assert_array_equal(td.feature_indices['P'], [0, 2, 4])
    np.testing.assert_array_equal(td.feature_indices['M'], [0, 1, 2])


def test_training_scheduler(four_omics):
    data, _, _, _, model = four_omics
    model, hist = flex.train_dtmae(model, data, n_epochs=2, batch_size=32,
        lr=0.001, step_size=1, scheduler_gamma=0.5, verbose=False)
    assert hist['lr'] == [0.001, 0.0005]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError, match='step_size'):
        flex.train_dtmae(model, data, n_epochs=1, step_size=0)


def make_flex(samples):
    data = flex.training_dataset(samples, ['G', 'R', 'P', 'M'],
        {'G': {'scale_factor': 0.5}, 'M': {'filter_zero_var': True},
         'P': {'keep_indices': [0, 2, 4]}})
    dims = {m: x.numel() for m, x in data[0][0].items()}
    hidden = {m: [8] for m in dims}
    model = flex.autoencoder(dims, ['G', 'M', 'P'], ['R'],
        dict(G=2, M=2, P=2), hidden, hidden, hidden,
        dict(G=['G', 'M', 'P'], M=['M', 'P'], P=['R', 'M', 'P']), 0)
    return data, model


def test_flexible_training_and_extraction(samples):
    data, model = make_flex(samples)
    model, hist = flex.train_dtmae(model, data, n_cells=8, time_per_cell=11,
        n_epochs=2, batch_size=32, lambda_manifold=0.01, verbose=False)
    assert len(hist['train_loss']) == 2
    assert np.isfinite(hist['val_loss']).all()
    scores = flex.evaluate_dtmae(model, data)
    assert np.isfinite(list(scores.values())).all()
    assert flex.extract_latent_embeddings(model, samples, data).shape == (96, 6)
    with pytest.raises(ValueError, match='T-1'):
        flex.train_dtmae(model, data, n_cells=8, time_per_cell=12, n_epochs=1)
    with pytest.raises(ValueError, match='observation'):
        flex.evaluate_dtmae(model, data, indices=[])


def test_splits_preserve_whole_cells_and_ragged_samples(samples):
    names = ['a'] * 3 + ['b'] * 5 + ['c'] * 2
    train, val = split_pairs(names, 0.3, 7)
    assert not set(np.array(names)[train]) & set(np.array(names)[val])
    assert sorted(np.r_[train, val].tolist()) == list(range(10))
    ragged = [tuple(t[:7] for t in samples[0][:4]) + samples[0][4:], samples[1]]
    data, model = make_flex(ragged)
    flex.train_dtmae(model, data, n_epochs=1, batch_size=16, verbose=False)
    with pytest.raises(ValueError, match='two cells'):
        split_pairs(['only'] * 5)
    with pytest.raises(ValueError, match='both'):
        split_pairs(names, n_cells=3)
    with pytest.raises(ValueError, match='val_fraction'):
        split_pairs(names, val_fraction=1)


def test_invalid_and_constant_inputs(samples, frames):
    with pytest.raises(ValueError, match='No cell'):
        cotree.OmicsDataset(None, ['group'], data={})
    with pytest.raises(ValueError, match='one cell'):
        flex.training_dataset([], ['G','R','P','M'])
    with pytest.raises(ValueError):
        flex.training_dataset(samples, ['G','R','P','M'], {'P': {'keep_indices': [-1]}})
    constant = [(s[0], s[1], s[2], torch.ones_like(s[3]), *s[4:]) for s in samples]
    with pytest.raises(ValueError, match='all features'):
        flex.training_dataset(constant, ['G','R','P','M'], {'M': {'filter_zero_var': True}})
    with pytest.raises(ValueError, match='all features'):
        flex.training_dataset([{'x': torch.ones(4, 3)}], ['x'], {'x': {'filter_zero_var': True}})
    with pytest.raises(ValueError, match='same time'):
        flex.training_dataset([{'a': torch.ones(4, 3), 'b': torch.ones(3, 2)}], ['a', 'b'])
    with pytest.raises(ValueError, match='80'):
        cotree.OmicsDataset(None, ['group'], deathfilter=True, data=frames)
    zeros = [(torch.zeros_like(s[0]), torch.zeros_like(s[1]), torch.zeros_like(s[2]), s[3], *s[4:]) for s in samples]
    td = flex.training_dataset(zeros, ['G','R','P','M'], {'P': {'keep_indices':[0]}})
    assert all(torch.isfinite(t).all() for batch in td[0] for t in batch.values())


def test_manifold_single_observation():
    z = torch.ones(1, 2, requires_grad=True)
    loss = cotree.graph_manifold_loss(torch.ones(1, 3), z)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize('width', [2, 4])
def test_preprocessing_rejects_incompatible_feature_widths(width):
    samples = [{'x': torch.ones(3, 3)}]
    incompatible = {'x': torch.ones(3, width)}
    config = {'x': {'keep_indices': [0, 1]}}
    with pytest.raises(ValueError, match='expected 3 raw features'):
        flex.training_dataset(samples + [incompatible], ['x'], config)
    data = flex.training_dataset(samples, ['x'], config)
    for transform in (lambda s: data.transform_samples([s]), data.preprocess_sample):
        with pytest.raises(ValueError, match='expected 3 raw features'):
            transform(incompatible)
    # Direct preprocessing also accepts NumPy input using the fitted schema.
    processed = data.preprocess_sample({'x': np.ones((3, 3))})
    torch.testing.assert_close(processed['x'], torch.ones(3, 2))


def test_custom_modalities_and_prediction_outputs():
    rng = torch.Generator().manual_seed(22)
    samples = [{'met': torch.rand(7, 3, generator=rng) + 1,
                'rna': torch.rand(7, 2, generator=rng) + 1, 'cell_id': i}
               for i in range(4)]
    data = flex.training_dataset(samples, ['met', 'rna'])
    # Predict a passthrough modality, not just encoded modalities.
    model = flex.autoencoder({'met':3,'rna':2}, ['met'], ['rna'], {'met':2},
        {'met':[4]}, {'met':[4]}, {'rna':[4]}, {'rna':['met','rna']}, dropout_rate=0)
    model, hist = flex.train_dtmae(model, data, n_epochs=1, verbose=False)
    assert 'pred_rna' in flex.evaluate_dtmae(model, data)
    assert np.isfinite(hist['val_loss']).all()
    assert flex.extract_latent_embeddings(model, samples, data).shape == (28,2)
