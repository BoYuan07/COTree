"""COTree: inference tools for multi-omics cell trajectory data.

"""
from ._version import __version__


from .driver_species_detection import extract_trajectory, driver_species_detection, plot_species_scores
from .DTMAE import OmicsDataset, training_dataset, train_autoencoder_model, RL_output, seperate_output, reconstruct_lineage
from .function_module_detection import function_module_estimation, functional_module_species, function_score_estimation
from .granger_causality_test import train_causality_model, plot_gc_heatmap
from .tree import construct_principal_tree, project_embeddings_to_tree, lineage_clustering, vis_tree, vis_predictive_time, vis_retrospective_time
