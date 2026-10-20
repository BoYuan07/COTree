"""Small CPU workflow. Run: python examples/quickstart.py --output-dir results.

The short training run demonstrates the API; it is not a fitted research model.
"""
# %% Imports and settings
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cotree
from cotree import dtmae_flex as flex

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', default='cotree-results')
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
device = torch.device('cpu')

# %% Create small positive four-omics trajectories
# CSV users can instead call OmicsDataset(data_folder, ['group_name']).
# The bundled data are at importlib.resources.files('cotree') / 'data'.
cell_num, time_length = 8, 24
t = np.linspace(0, 4, time_length)
data = {}
for i in range(cell_num):
    sign = -1 if i < cell_num // 2 else 1
    branch = sign * np.maximum(t - 1.5, 0)
    frames = {}
    for modality, n_features in [('genomics', 5), ('transcriptomics', 3),
                                  ('proteomics', 6), ('metabolomics', 5)]:
        slope = np.linspace(0.4, 1.4, n_features)
        effect = np.linspace(-0.3, 0.3, n_features)
        values = 5 + t[:, None]*slope + branch[:, None]*effect
        values += rng.normal(0, 0.02, values.shape)
        if modality == 'metabolomics':
            values[:, -1] = 2  # one invariant species, removed during preprocessing
        names = [f'{modality}_{j}' for j in range(n_features)]
        frames[modality] = pd.DataFrame(values.T, index=names)
    data[f'group{i}'] = frames

dataset = cotree.OmicsDataset(None, ['group'], data=data)
keep_protein = list(range(len(dataset.names['proteomics'])))
training = flex.training_dataset(dataset.samples, ['G','R','P','M'], {
    'G': {'scale_factor': 0.5}, 'M': {'filter_zero_var': True},
    'P': {'keep_indices': keep_protein},
})
input_size = {m: x.numel() for m,x in training[0][0].items()}
hidden = {m: [12] for m in 'GMPR'}
latent_dim = {'G': 4, 'M': 4, 'P': 4}

# %% Train the current configurable model and extract embeddings
predictor_inputs = {'G': ['G', 'M', 'P'], 'M': ['M', 'P'], 'P': ['R', 'M', 'P']}
model = flex.autoencoder(
    input_size, ['G','M','P'], ['R'], latent_dim,
    hidden, hidden, hidden, predictor_inputs, dropout_rate=0, normalize_input=['M','P'],
)
model, history = flex.train_dtmae(
    model, training, n_epochs=args.epochs, batch_size=64,
    val_fraction=0.25, seed=42, device=device, verbose=False,
)
processed = training.transform_samples(dataset.samples)
Z = flex.extract_latent_embeddings(model, dataset.samples, training).reshape(cell_num, time_length, -1)
# A common root is used for tree construction. Keep original Z for model analyses.
tree_embeddings = (Z - Z[:, :1]).reshape(cell_num*time_length, -1)

# %% Fit trajectories and infer times (five returned values)
trajectory_tree, stages, nodes, bifurcations, distances = cotree.construct_principal_tree(
    tree_embeddings, time_length, cell_num,
)
labels = cotree.lineage_clustering(distances, time_length, trajectory_tree)
cotree.vis_tree(tree_embeddings, trajectory_tree, np.repeat(labels, time_length),
                nodes, '(example)', save_path=output_dir/'tree.png')
cotree.vis_tree_temporal(tree_embeddings, trajectory_tree, np.tile(t, cell_num),
                        bifurcation_point=nodes, save_path=output_dir/'tree-time.png')
cotree.vis_predictive_time(trajectory_tree, distances, nodes, labels, time_length,
                          save_path=output_dir/'predictive-time.png')
cotree.vis_retrospective_time(trajectory_tree, stages, bifurcations, labels, time_length,
                             save_path=output_dir/'retrospective-time.png')

# %% Functional module and species importance using the same flexible model
v = cotree.function_module_estimation(model, training, 0, 0.01, input_size, device, epochs=2)
G, M, P = [processed[m] for m in 'GMP']
tau, module_species, _, _ = cotree.functional_module_species(G, M, P, v, model, 0, input_size, device)
importance = cotree.function_score_estimation(model, 0, dataset, training, device=device)
np.save(output_dir/'functional-importance.npy', importance)

# %% Driver-species inputs and optional detection at the first supported split
raw = torch.cat([processed[m] for m in 'GMP'], dim=-1).numpy()
raw_by_cell = raw.reshape(cell_num, time_length, -1)
# Use unshifted model embeddings for driver analysis; use tree stages for alignment.
raw_nodes, encoded_nodes, weights, real_time, lineage_map = cotree.extract_trajectory(
    trajectory_tree, stages.reshape(cell_num, time_length, -1), raw_by_cell, Z,
    labels, bifurcations, list_truncation=time_length, inter=1,
)
driver_species = []
for node in raw_nodes:
    if all(np.asarray(x).ndim == 2 and len(x) >= 2 for x in raw_nodes[node].values()):
        coefficients, selected = cotree.driver_species_detection(
            model, node, raw_nodes, encoded_nodes, weights, input_size, device,
            lineage_map=lineage_map, epoches=2, threshold=1e6, n_grid=12,
        )
        driver_species = selected.tolist()
        break

# %% Evaluate the trained model without fitting a second model
scores = flex.evaluate_dtmae(model, training)

# %% Save a compact result summary
summary = {
    'version': cotree.__version__, 'cells': cell_num, 'time_points': time_length,
    'lineages': len(trajectory_tree), 'embedding_shape': list(tree_embeddings.shape),
    'functional_species': len(module_species), 'driver_species': driver_species,
    'validation_loss': history['val_loss'][-1], 'model_loss': scores['loss'],
}
(output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
plt.close('all')
print(json.dumps(summary, indent=2))
