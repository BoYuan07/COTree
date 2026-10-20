import importlib
import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
import cotree

tree = importlib.import_module('cotree.tree')
driver = importlib.import_module('cotree.driver_species_detection')


def test_branching_tree_and_plots(branch_embeddings, tmp_path):
    X, C, T = branch_embeddings
    fitted, stages, nodes, bif, distances = cotree.construct_principal_tree(
        X, T, C, n_before=5, K_max=3, K_min=2, max_iter=2,
        sample_num=20, epsilon=0.01, alpha_range=[0, 0.1, 2])
    K = len(fitted)
    assert K >= 2
    assert stages.shape == distances.shape == (C*T, K)
    assert nodes.ndim == 2
    assert list(fitted) == list(bif)
    assert np.all(np.diff(stages.reshape(C, T, K), axis=1) >= -1e-9)
    assert np.isfinite(distances).all()
    labels = cotree.lineage_clustering(distances, T, fitted)
    assert labels.shape == (C,)
    cotree.vis_tree(X, fitted, np.repeat(labels, T), nodes, '(test)', save_path=tmp_path/'tree.png')
    cotree.vis_predictive_time(fitted, distances, nodes, labels, T, save_path=tmp_path/'predictive.png')
    cotree.vis_retrospective_time(fitted, stages, bif, labels, T, save_path=tmp_path/'retrospective.png')
    assert (tmp_path/'tree.png').stat().st_size > 0
    plt.close('all')


def test_single_lineage_and_absent_crossings(tmp_path):
    t = np.linspace(0, 2, 10)
    X = np.tile(np.column_stack([t, t*t]), (4, 1))
    fitted, stages, nodes, bif, dist = cotree.construct_principal_tree(
        X, 10, 4, n_before=3, K_min=1, K_max=1, max_iter=1,
        epsilon=0.01, sample_num=12, alpha_range=[0, 0, 1])
    assert nodes.shape == (1, 2)
    assert stages.shape == dist.shape == (40, 1)
    labels = np.repeat('Lineage1', 4)
    pred = cotree.vis_predictive_time(fitted, dist, nodes, labels, 10, tmp_path/'one.png')
    assert pred['Lineage1'] == [0]*4
    retro = cotree.vis_retrospective_time(fitted, stages, {'Lineage1': 1e6}, labels, 10, tmp_path/'no-cross.png')
    assert np.isnan(retro['Lineage1']).all()
    plt.close('all')


def test_convergence_and_degenerate_projection():
    t = np.linspace(0, 5, 40)
    bend = np.sin(np.pi*t/5)**2
    curves = {'A': np.c_[t, bend], 'B': np.c_[t, -bend]}
    nodes, merged, times, bif = tree.detect_merge_bifurcations(
        curves, [t, t], abs_epsilon=0.02, allow_convergence=True)
    assert nodes.ndim == 2
    assert np.linalg.norm(merged['A'][-1]-merged['B'][-1]) < 1e-8
    X = np.ones((6, 2))
    stages, d, weights, seg = tree.project_embeddings_to_tree(X, {'Lineage1': np.zeros((2, 2))}, 3)
    assert np.all(stages == 0) and np.allclose(d, 2) and np.allclose(weights, 1)
    with pytest.raises(ValueError, match='divide'):
        tree.project_embeddings_to_tree(X, {'Lineage1': X}, 4)


def test_functional_analysis(four_omics, frames):
    data, dims, latent, hidden, model = four_omics
    v = cotree.function_module_estimation(model, data, 0, 0.01, dims, 'cpu', epochs=1)
    assert v.shape == (sum(dims[m] for m in 'GMP'),)
    G, M, P = [torch.stack([s[0][m] for s in data]) for m in 'GMP']
    for index in (0, [0]):
        tau, selected, diff, grid = cotree.functional_module_species(G, M, P, np.ones_like(v), model, index, dims, 'cpu')
        assert len(selected) == len(v)
    ds = cotree.OmicsDataset(None, ['group'], data=frames)
    scores = cotree.function_score_estimation(model, 0, ds, data)
    assert scores.shape == v.shape and np.isfinite(scores).all()


def test_driver_high_node_index_and_empty_direction(four_omics):
    data, dims, latent, hidden, model = four_omics
    X = torch.stack([torch.cat([data[i][0][m] for m in 'GMP']) for i in range(8)])
    with torch.no_grad():
        Z = model.encoder(dict(zip('GMP', cotree.seperate_output(X, dims)[:3])))[1]
    raw = {3: {'direction 1': X.numpy(), 'direction 2': (X * 1.05).numpy()}}
    encoded = {3: {'direction 1': Z.numpy(), 'direction 2': Z.numpy()}}
    weights = {3: {k: np.linspace(0.01, 0.2, 8) for k in raw[3]}}
    original = {k: v.copy() for k, v in weights[3].items()}
    b, selected = cotree.driver_species_detection(model, 3, raw, encoded, weights, dims, 'cpu', epoches=2, n_grid=8)
    assert b.shape == (10,) and torch.isfinite(b).all()
    for k in original:
        np.testing.assert_array_equal(original[k], weights[3][k])
    ax = cotree.plot_species_scores(b, [f's{i}' for i in range(10)], {3: {'direction 1':'A','direction 2':['B']}}, 3, selected)
    assert ax is not None
    raw[3]['direction 2'] = np.empty((0, 10))
    with pytest.raises(ValueError, match='two observations'):
        cotree.driver_species_detection(model, 3, raw, encoded, weights, dims, 'cpu', epoches=1)
    plt.close('all')


def test_graph_refinement_with_convergence():
    t = np.linspace(0, 4, 16)
    curves = [np.c_[t, sign * np.sin(np.pi*t/4)**2, t*t/10]
              for sign in (-1, 1) for _ in range(3)]
    out = cotree.construct_principal_tree(np.vstack(curves), 16, 6,
        n_before=4, K_max=2, K_min=1, use_tree=False, allow_nontree=True,
        max_iter=2, sample_num=18, alpha_range=[0, 0, 1])
    assert out[1].shape == out[4].shape
    assert np.isfinite(out[1]).all() and np.isfinite(out[4]).all()
    assert out[2].ndim == 2


def test_general_time_lengths_in_new_plots(branch_embeddings, tmp_path):
    X, C, T = branch_embeddings
    curves = {'Lineage1': X[:T], 'Lineage2': X[-T:]}
    labels = np.array(['Lineage1']*4 + ['Lineage2']*4)
    nodes = X[[0, 6]]
    ax = cotree.vis_tree_per_lineage(X, curves, labels, nodes, 1,
        save_path=tmp_path/'lineage.png', time_length=T)
    assert ax is not None
    stages = np.tile(np.linspace(0, 4, T)[None, :, None], (C, 1, 2))
    cotree.plot_differential_species(X.reshape(C,T,3), stages, ['x', 'y', 'z'],
        {'Lineage1':1.0,'Lineage2':1.0},
        {0:{'direction 1':'Lineage1','direction 2':['Lineage2']}}, [1], labels,
        save_dir=tmp_path, node=0)
    assert (tmp_path/'y_node0.jpeg').stat().st_size > 0
    plt.close('all')


def test_extract_trajectory_has_no_empty_terminal_comparison(branch_embeddings):
    X, C, T = branch_embeddings
    curves = {'Lineage1': X[:T], 'Lineage2': X[-T:]}
    labels = np.array(['Lineage1']*4 + ['Lineage2']*4)
    stages = np.tile(np.linspace(0,4,T)[None,:,None], (C,1,2))
    raw, encoded, weights, times, mapping = cotree.extract_trajectory(
        curves, stages, X.reshape(C,T,3), X.reshape(C,T,3), labels,
        {'Lineage1':1.0,'Lineage2':1.0}, list_truncation=T, inter=1)
    assert list(raw) == [0]
    assert mapping[0]['direction 2'] == ['Lineage2']
    assert all(v.ndim == 2 and len(v) > 1 for v in raw[0].values())


def test_importance_uses_model_latent_order_and_fitted_masks(samples, frames):
    from cotree import dtmae_flex as flex
    data = flex.training_dataset(samples, ['G','R','P','M'], {
        'M': {'filter_zero_var': True},
        'P': {'keep_indices': [4,2,0], 'filter_zero_var': True},
    })
    ds = cotree.OmicsDataset(None, ['group'], data=frames)
    inputs = data.transform_samples(ds.samples)
    dims = {m: v.shape[1] for m,v in inputs.items()}
    hidden = {m:[8] for m in dims}
    model = flex.autoencoder(dims, ['P','G','M'], ['R'], dict(G=2,M=2,P=2),
        hidden, hidden, hidden, dict(G=['G','M','P'],M=['M','P'],P=['R','M','P']),0)
    score = cotree.function_score_estimation(model, 0, ds, data)
    with torch.no_grad():
        embeddings, _, _, prediction = model(inputs)
        embeddings['P'][:,0] = 0  # first latent coordinate belongs to P, not G
        perturbed = model.predictor(embeddings, {'R':inputs['R']})
        X = torch.cat([prediction[m] for m in 'GMP'],dim=-1)
        Y = torch.cat([perturbed[m] for m in 'GMP'],dim=-1)
        expected = (Y-X).norm(dim=0)/(X.norm(dim=0)+1e-8)
    np.testing.assert_allclose(score, expected, atol=1e-7)
    np.testing.assert_array_equal(data.feature_indices['P'], [4,2,0])
    with pytest.raises(ValueError, match='encoded modalities'):
        bad_model = flex.autoencoder({'G':4}, ['G'], [], {'G':2}, {'G':[8]},
            {'G':[8]}, {'G':[8]}, {'G':['G']},0)
        cotree.functional_module_species(inputs['G'], inputs['M'], inputs['P'],
            np.ones(sum(dims[m] for m in 'GMP')), bad_model, 0, dims, 'cpu')
