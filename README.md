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
(e.g. the `gnn-env` conda env) is needed to run the GNN methods. The GNN
methods support two graph scopes, selected via `graph_scope` in the params TSV
(see "Graph scope" below). Install PyTorch separately after creating the
environment.

**Already running the pipeline (baselines/XGBoost) and just adding the GNNs?**
You only need to add two packages to your existing pipeline environment — the
data paths, slurm profile and `gpu_p` GPU path you already use for XGBoost are
unchanged:

```bash
# into your existing env; pick the cuXXX matching your cluster CUDA driver
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.7.0   # pure-Python; no torch-scatter/sparse needed
```

Reference working versions: `torch==2.5.1+cu121`, `torch_geometric==2.7.0`. If
these are missing, only the `gcn`/`graphsage` jobs fail (with a clear
ImportError); with `keep-going: True` in the slurm profile every non-GNN job
and the report still complete.

Validation is carved from the train nodes via `val_strategy` (in the GNN
params TSVs): `spatial_band` (default) holds out a contiguous coordinate band
(the `val_fraction` slab along `val_axis`, where `val_axis="auto"` picks the
axis of largest spatial extent), which keeps the induced validation subgraph
connected so early stopping has a meaningful signal; `random` reproduces a
random hold-out for comparison. For a spatial test split (e.g.
`split_spatial_y_median`), setting `val_axis` to that same axis makes the
validation hold-out a closer proxy for the test distribution shift.

#### Graph construction

The GNN methods build a spatial graph whose relationship to the train/test
split is set by `graph_scope` (see "Graph scope" below): `within_split` builds
an independent graph per slice (no train-test edges), while `transductive`
builds one graph over all nodes. The graph itself is controlled by
`graph_source` in the GNN params TSVs:

- `obsp` — reuse a precomputed connectivity matrix from `adata.obsp[obsp_key]`
  (default `obsp_key = spatial_connectivities`). In `within_split` scope, when
  AnnData is sliced into train/test the cross-split edges are dropped
  automatically so each slice keeps only its within-split connectivity; in
  `transductive` scope the full connectivity is used.
- `radius_capped_knn` — rebuild the graph from the spatial coordinates in
  `adata.obsm['spatial']`. A `knn_k`-nearest-neighbor graph is capped by a
  distance threshold so long, spurious edges across gaps are removed. The
  threshold is set by `radius_strategy` (`kth_neighbor_percentile`): take the
  `radius_percentile` of each node's k-th neighbor distance, then allow edges
  up to `max_radius_multiplier ×` that radius. If `repair_isolates = 1`, any
  node left with no edges is reconnected to its nearest neighbor so the graph
  has no isolated nodes.

For each run, the resolved graph is profiled and the QC statistics are written
into `run_metadata.json` under `model_metadata` (`n_nodes`, `n_edges`,
`mean_degree`, `isolated_nodes`, `n_components`, `giant_component_fraction`,
`median_edge_length`), so graph health can be inspected alongside the metrics.
The QC is keyed by the train/test graphs for `within_split`, and by
`full`/`train_fit`/`val`/`test` for `transductive` (which also records
`frac_test_nodes_with_train_neighbor`).

#### GNN hyperparameters

The GNN param grids are **split-specific**, since the right configuration
depends on the split: use `params/gcn_params_random.tsv` /
`params/graphsage_params_random.tsv` for random splits (these hold
`transductive` rows with `val_strategy=random` plus a `within_split` ablation),
and `params/gcn_params_spatial.tsv` / `params/graphsage_params_spatial.tsv` for
spatial holdout splits (`within_split` / inductive, `val_strategy=spatial_band`,
with diversified graph construction). The remaining columns in these files
configure the model and training loop:
`hidden_dim`, `num_layers`, `dropout`, `lr`, `weight_decay`, `epochs`,
`patience` (early-stopping patience on validation loss), `standardize` (fit a
`StandardScaler` on the train-fit nodes only), and `seed`. GraphSAGE adds
`aggr` (neighbor aggregation, e.g. `mean` or `max`). Models are trained with
MSE loss and the Adam optimizer on GPU when available, otherwise CPU.

#### Graph scope (inductive vs transductive)

`graph_scope` selects how the graph relates to the train/test split:

- `within_split` (default, inductive): the train graph is built from the train
  slice and the test graph from the test slice, independently, with no
  train-test edges. The model never aggregates from labeled train neighbors at
  inference. This is the right choice for spatial holdout splits, where train
  and test cover different tissue regions.
- `transductive`: a single graph is built over all nodes (train + test). The
  training loss is masked to the train-fit nodes, validation/early-stopping
  uses train-only held-out nodes, and test predictions are read out from nodes
  that can aggregate messages from their labeled train neighbors. This is the
  setting where a GNN can exploit neighborhood structure on a non-spatial
  (e.g. random 80/20) split, which `within_split` structurally cannot do.

Important caveat for `transductive`: test-node *features* (RNA) participate in
message passing during training, although test-node *labels* (MSI) are never
used for the loss, validation, early stopping or model selection. This is
standard semi-supervised GNN practice, but it is a different evaluation regime
than the inductive baselines (linear / XGBoost / `within_split` GNN), which
never see test features. Runs are tagged in `run_metadata.json` with
`uses_test_features_in_message_passing` and
`frac_test_nodes_with_train_neighbor` so the regime is explicit; to isolate the
value of transduction, run both scopes on the same split and compare. Note that
`transductive` requires the unsliced AnnData, which `run_method.py` passes
automatically for the GNN methods.

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
cp config_example.yaml config.yaml
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

The provided `profile_slurm/config.yaml` is **cluster-specific**: before using
it, edit the partition / QoS / GRES names (`partition`, `qos`, `gres`), the
`--exclude` node list, and the per-rule resources in `snakefile` (the
`GPU_METHODS` set routes `xgboost`/`gcn`/`graphsage` to the GPU partition) to
match your cluster. After a code change, snakemake re-runs all jobs by default;
add `--rerun-triggers mtime` to reuse existing results and only compute new
runs.

---

## Notes

Input `.h5ad` files are not included in this repository.

The pipeline assumes that the input and target AnnData objects are already aligned. It will not reorder observations automatically.
