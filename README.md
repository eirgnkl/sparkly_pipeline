# sparkly_pipeline ✨

`sparkly_pipeline` is a Snakemake pipeline for benchmarking cross-modal prediction between spatial omics modalities.

The main use case is predicting **MSI / metabolite intensities from RNA / gene expression**, but the pipeline can also be used in the opposite direction by switching the input files in the config.

The pipeline currently supports:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- XGBoost
- GCN (graph neural network; requires torch + torch_geometric)
- GraphSAGE (graph neural network; requires torch + torch_geometric)

It trains the selected models, computes global and per-metabolite metrics, merges the results, selects the best models, and generates an HTML report.

---

## Input data

The pipeline expects two prepared `.h5ad` files:

- one AnnData file used as input features
- one AnnData file used as prediction target

The files should already be preprocessed and aligned before running the pipeline.

Requirements:

- both AnnData objects must contain the same observations
- observation order must match
- feature values should be stored in `.X`
- feature names should be stored in `.var_names`
- the train/test split column must exist in `.obs`
- split labels should be `"train"` and `"test"`

The pipeline does **not** perform preprocessing, normalization, alignment, or feature selection internally.

---

## Installation

Create the environment:

```bash
mamba env create -f env.yaml
mamba activate pipeline
```

or with conda:

```bash
conda env create -f env.yaml
conda activate pipeline
```

The environment includes the main dependencies for the baseline models, including Snakemake, Scanpy, scikit-learn, XGBoost, CuPy, and Plotly.

### GPU note

This pipeline assumes that the user has a GPU-capable workstation.

To check whether the GPU is visible:

```bash
nvidia-smi
```

To check CuPy:

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

### Optional GNN dependencies

The baseline (non-GNN) models do not require PyTorch. The `gcn` and
`graphsage` methods do: torch and torch_geometric are imported lazily, so the
other methods keep working without them, but a torch-capable environment
(e.g. the `gnn-env` conda env) is needed to run the GNN methods. These methods
currently support only within-split graph learning (train graph from the train
slice, validation drawn only from train, test graph from the test slice; no
train-test edges). Install PyTorch separately after creating the environment.

Validation is carved from the train nodes via `val_strategy` (in the GNN
params TSVs): `spatial_band` (default) holds out a contiguous coordinate band
(the `val_fraction` slab along `val_axis`, where `val_axis="auto"` picks the
axis of largest spatial extent), which keeps the induced validation subgraph
connected so early stopping has a meaningful signal; `random` reproduces a
random hold-out for comparison. For a spatial test split (e.g.
`split_spatial_y_median`), setting `val_axis` to that same axis makes the
validation hold-out a closer proxy for the test distribution shift.

#### Graph construction

The GNN methods build a spatial graph **per split** (within-split scope), so
there are never edges between train and test nodes. The graph is controlled
by `graph_source` in the GNN params TSVs:

- `obsp` — reuse a precomputed connectivity matrix from `adata.obsp[obsp_key]`
  (default `obsp_key = spatial_connectivities`). When AnnData is sliced into
  train/test, cross-split edges are dropped automatically, so each slice keeps
  only its within-split connectivity.
- `radius_capped_knn` — rebuild the graph from the spatial coordinates in
  `adata.obsm['spatial']`. A `knn_k`-nearest-neighbor graph is capped by a
  distance threshold so long, spurious edges across gaps are removed. The
  threshold is set by `radius_strategy` (`kth_neighbor_percentile`): take the
  `radius_percentile` of each node's k-th neighbor distance, then allow edges
  up to `max_radius_multiplier ×` that radius. If `repair_isolates = 1`, any
  node left with no edges is reconnected to its nearest neighbor so the graph
  has no isolated nodes.

For each run, the resolved graph is profiled and the QC statistics are written
into `run_metadata.json` under `model_metadata` (per split: `n_nodes`,
`n_edges`, `mean_degree`, `isolated_nodes`, `n_components`,
`giant_component_fraction`, `median_edge_length`), so graph health can be
inspected alongside the metrics.

#### GNN hyperparameters

The remaining columns in `params/gcn_params.tsv` and
`params/graphsage_params.tsv` configure the model and training loop:
`hidden_dim`, `num_layers`, `dropout`, `lr`, `weight_decay`, `epochs`,
`patience` (early-stopping patience on validation loss), `standardize` (fit a
`StandardScaler` on the train-fit nodes only), and `seed`. GraphSAGE adds
`aggr` (neighbor aggregation, e.g. `mean` or `max`). Models are trained with
MSE loss and the Adam optimizer on GPU when available, otherwise CPU.

For example, for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
```

Check PyTorch GPU access:

```bash
python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## Configuration

Copy the example config:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` with the paths to your local `.h5ad` files.

Example:

```yaml
TASKS:
  rna_to_msi_example:
    input_rna: /path/to/prepared_rna.h5ad
    input_metabolomics: /path/to/prepared_msi.h5ad
    split: split_random_80_20
    methods:
      ridge:
        params: params/ridge_params.tsv
      lasso:
        params: params/lasso_params.tsv
      linear:
        params: params/linreg_params.tsv
      elastic_net:
        params: params/elastic_net_params.tsv
      xgboost:
        params: params/xgboost_params.tsv
```

The names `input_rna` and `input_metabolomics` are historical. In practice:

- `input_rna` is the input feature matrix
- `input_metabolomics` is the prediction target

To run MSI → RNA, switch the file paths.

---

## Running locally

First check the planned jobs, via dry run:

```bash
snakemake -n -p
```

Then run the pipeline:

```bash
snakemake --cores 8
```

Adjust the number of cores depending on your workstation.

---

## Outputs

All results are written to:

```text
data/reports/{task}/
```

Each individual model run is stored under:

```text
data/reports/{task}/{model}/{hash}/
```

with files such as:

```text
global_metrics.tsv
per_metabolite_metrics.parquet
run_metadata.json
```

The task-level outputs include:

```text
merged_global_metrics.tsv
merged_per_metabolite_metrics.parquet
best_models.tsv
{task}_model_report.html
```

The main file to open after the run is:

```text
data/reports/{task}/{task}_model_report.html
```

This report summarizes model performance and selected top models.

---

## Running on a cluster

This repository also contains a Slurm profile, but this is optional.

For local workstation use, prefer:

```bash
snakemake --cores 8
```

For Slurm usage:

```bash
snakemake --profile profile_slurm
```

---

## Notes

Input `.h5ad` files are not included in this repository.

The pipeline assumes that the input and target AnnData objects are already aligned. It will not reorder observations automatically.
