import importlib.util
from importlib.metadata import metadata
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import cotree
from packaging.requirements import Requirement


def test_public_api_and_removed_module():
    assert cotree.__version__ == '0.5.0'
    from cotree import dtmae_flex as flex
    import cotree.DTMAE as helpers
    for name in ('autoencoder', 'training_dataset', 'vis_pca'):
        assert getattr(cotree, name) is getattr(flex, name)
        assert not hasattr(helpers, name)
    for name in ('train_autoencoder_model', 'RL_output', 'stacked_dataset'):
        assert not hasattr(cotree, name) and not hasattr(helpers, name)
    for name in ('PCAAE', 'UMAPAE', 'TSNEAE', 'MOFAAE', 'Accuracy_comparison',
                 'vis_other_rep', 'ClassificationAccuracy_logistic', 'logistic_regression'):
        assert all(not hasattr(module, name) for module in (cotree, helpers, flex))

    assert all(hasattr(cotree, name) for name in cotree.__all__)
    assert not hasattr(cotree, 'train_causality_model')
    assert not hasattr(cotree, 'plot_gc_heatmap')
    assert importlib.util.find_spec('cotree.granger_causality_test') is None


def test_tuning_is_optional_and_removed_libraries_are_not_imported():
    script = '''
import sys
class DenyOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'umap', 'mofapy2', 'optuna'}:
            raise ImportError('Deliberately unavailable optional dependency')
sys.meta_path.insert(0, DenyOptional())
import cotree
try: cotree.tune_dtmae(None, {}, [], [], {})
except ImportError as exc: assert 'tuning' in str(exc), str(exc)
else: raise AssertionError('Tuning did not explain its missing dependency')
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_reaction_dependencies_are_standard_and_comparisons_removed():
    package = metadata('cotree')
    requirements = [Requirement(r) for r in package.get_all('Requires-Dist', [])]
    required = {r.name.lower() for r in requirements if r.marker is None}
    assert {'cobra', 'networkx'} <= required
    assert not {'umap-learn', 'mofapy2'} & {r.name.lower() for r in requirements}
    assert set(package.get_all('Provides-Extra', [])) == {'tuning', 'dev'}


def test_tuning(samples):
    pytest.importorskip('optuna')
    from test_models import make_flex
    data, model = make_flex(samples)
    dims = {m: x.numel() for m, x in data[0][0].items()}
    best = cotree.tune_dtmae(data, dims, ['G', 'M', 'P'], ['R'],
        dict(G=['G','M','P'], M=['M','P'], P=['R','M','P']),
        n_cells=8, time_per_cell=11, n_trials=1, n_epochs_per_trial=1,
        search_space=dict(hidden_choices=[8], latent_G=[2], latent_M=[2], latent_P=[2], batch_size=32),
        verbose=False)
    assert best['latent_dims'] == dict(G=2, M=2, P=2)
    assert np.isfinite(best['eta_curr'])


def test_reaction_plot_uses_supplied_file(tmp_path, monkeypatch):
    # Keep COBRA's import-time cache inside this test's temporary directory.
    import appdirs
    import matplotlib.pyplot as plt
    monkeypatch.setattr(appdirs, 'user_cache_dir', lambda *a, **k: str(tmp_path/'cache'))
    monkeypatch.setattr(plt, 'show', lambda: None)
    import cobra
    model = cobra.Model('small')
    a, b = cobra.Metabolite('a_c', compartment='c'), cobra.Metabolite('b_c', compartment='c')
    reaction = cobra.Reaction('R1')
    reaction.add_metabolites({a: -1, b: 1})
    model.add_reactions([reaction])
    sbml = tmp_path/'model.xml'
    cobra.io.write_sbml_model(model, str(sbml))
    output = tmp_path/'reaction.png'
    cotree.plot_metabolite_reaction(1, {'Functional Module 1':['M_a_c','M_b_c']},
        output, {'Functional Module 1':[0.1, -0.2]}, sbml_path=sbml)
    assert output.stat().st_size > 0
    assert plt.gcf()._suptitle.get_text() == 'Metabolite Module 1'
    plt.close('all')


def test_tutorial_uses_defaults_and_does_not_execute_tuning():
    import ast
    path = Path(__file__).resolve().parents[1] / 'COTree.ipynb'
    if not path.exists():
        pytest.skip('Tutorial source not available')
    notebook = json.loads(path.read_text())
    tree_calls = []
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code':
            continue
        syntax = ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
        for node in ast.walk(syntax):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                assert name != 'tune_dtmae'
                assert name != 'train_autoencoder_model'
                if name == 'construct_principal_tree':
                    tree_calls.append(node)
    assert len(tree_calls) == 1 and not tree_calls[0].keywords
    example = next(''.join(c['source']) for c in notebook['cells']
                   if c['cell_type']=='markdown' and 'best = flex.tune_dtmae' in ''.join(c['source']))
    ast.parse(example.split('```python\n',1)[1].split('```',1)[0])
