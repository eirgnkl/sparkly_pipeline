# sparkly_pipeline ✨

`sparkly_pipeline` is a Snakemake pipeline for benchmarking cross-modal prediction between spatial omics modalities.

The main use case is predicting **MSI / metabolite intensities from RNA / gene expression**, but the pipeline can also be used in the opposite direction by switching the input files in the config.

The pipeline currently supports:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- XGBoost

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

The current baseline pipeline does not require PyTorch. For future GNN/cVAE development, install PyTorch separately after creating the environment.

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
model_report.html
```

The main file to open after the run is:

```text
data/reports/{task}/model_report.html
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
