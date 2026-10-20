"""COTree: inference for cell-resolved multi-omics trajectories."""
from ._version import __version__
from .DTMAE import (
    OmicsDataset, seperate_output, reconstruct_lineage,
    cell_trajectory_val_split, graph_manifold_loss,
)
from .driver_species_detection import (
    extract_trajectory, driver_species_detection, plot_species_scores,
    plot_differential_species,
)
from .function_module_detection import (
    function_module_estimation, functional_module_species,
    function_score_estimation, plot_metabolite_reaction,
)
from .tree import (
    construct_principal_tree, project_embeddings_to_tree, lineage_clustering,
    vis_tree, vis_tree_temporal, vis_tree_per_lineage,
    vis_predictive_time, vis_retrospective_time,
)
from . import dtmae_flex
from .dtmae_flex import (
    autoencoder, training_dataset, vis_pca,
    train_dtmae, evaluate_dtmae, extract_latent_embeddings,
    tune_dtmae, plot_training_history,
)

__all__ = [
    '__version__', 'OmicsDataset', 'training_dataset', 'autoencoder',
    'seperate_output',
    'reconstruct_lineage', 'cell_trajectory_val_split', 'graph_manifold_loss',
    'vis_pca',
    'extract_trajectory', 'driver_species_detection', 'plot_species_scores',
    'plot_differential_species', 'function_module_estimation',
    'functional_module_species', 'function_score_estimation',
    'plot_metabolite_reaction', 'construct_principal_tree',
    'project_embeddings_to_tree', 'lineage_clustering', 'vis_tree',
    'vis_tree_temporal', 'vis_tree_per_lineage', 'vis_predictive_time',
    'vis_retrospective_time', 'dtmae_flex', 'train_dtmae', 'evaluate_dtmae',
    'extract_latent_embeddings', 'tune_dtmae', 'plot_training_history',
]
