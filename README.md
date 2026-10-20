# COTree

This is a Python implementation of the COTree framework developed for multi-omics cell trajectory data analysis. See the [paper](https://www.biorxiv.org/content/10.1101/2025.10.30.685684v1).

A complete working example is located in [COTree.ipynb](COTree.ipynb).

## Install

Python 3.9 or later and Git are required. Install the package directly from GitHub, including Optuna for hyperparameter tuning:

```bash
python -m pip install "cotree[tuning] @ git+https://github.com/BoYuan07/COTree.git"
```

## Usage

```python
import cotree
print(cotree.__version__)
```

See the [tutorial](COTree.ipynb) for the analysis workflow.
