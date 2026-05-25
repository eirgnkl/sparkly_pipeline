import ast
import json
from pathlib import Path

import scanpy as sc

from ridge import run_ridge_reg
from linear import run_linreg
from lasso import run_lasso
from mxgboost import run_xgboost
from elastic_net import run_elastic_net
from model_utils import (
    compute_global_metrics,
    compute_per_metabolite_metrics,
    save_json,
    save_predictions_npz,
)


METHOD_MAP = {
    "ridge": dict(function=run_ridge_reg, mode="paired"),
    "lasso": dict(function=run_lasso, mode="paired"),
    "linear": dict(function=run_linreg, mode="paired"),
    "xgboost": dict(function=run_xgboost, mode="paired"),
    "elastic_net": dict(function=run_elastic_net, mode="paired"),
}


params = snakemake.params.thisparam
rna_path = snakemake.input.rna_ds
metab_path = snakemake.input.metab_ds

method = str(params["method"]).strip()
task = str(params["task"]).strip()
hash_id = str(params["hash"]).strip()
split_name = str(params.get("split", "split")).strip()
rna_layer = str(params.get("rna_layer", "X")).strip()
msi_layer = str(params.get("msi_layer", "X")).strip()

if method not in METHOD_MAP:
    raise ValueError(
        f"Unknown method '{method}'. Available methods: {list(METHOD_MAP.keys())}"
    )

raw_method_params = params.get("params", "{}")
if isinstance(raw_method_params, dict):
    method_params = raw_method_params
else:
    method_params = ast.literal_eval(raw_method_params)

method_function = METHOD_MAP[method]["function"]

adata_rna = sc.read_h5ad(rna_path)
adata_msi = sc.read_h5ad(metab_path)

if adata_rna.n_obs != adata_msi.n_obs:
    raise ValueError(
        f"RNA/MSI n_obs mismatch: RNA has {adata_rna.n_obs}, MSI has {adata_msi.n_obs}"
    )

if not (adata_rna.obs_names == adata_msi.obs_names).all():
    raise ValueError("RNA and MSI obs_names are not aligned. Refusing to reorder automatically.")

if split_name not in adata_rna.obs.columns:
    raise KeyError(
        f"Split column '{split_name}' not found in RNA obs. "
        f"Available columns: {list(adata_rna.obs.columns)}"
    )

if split_name not in adata_msi.obs.columns:
    raise KeyError(
        f"Split column '{split_name}' not found in MSI obs. "
        f"Available columns: {list(adata_msi.obs.columns)}"
    )

train_mask = adata_rna.obs[split_name].astype(str).eq("train").to_numpy()
test_mask = adata_rna.obs[split_name].astype(str).eq("test").to_numpy()

if train_mask.sum() == 0:
    raise ValueError(f"No train observations found in split column '{split_name}'.")
if test_mask.sum() == 0:
    raise ValueError(f"No test observations found in split column '{split_name}'.")

adata_rna_train = adata_rna[train_mask, :]
adata_rna_test = adata_rna[test_mask, :]
adata_msi_train = adata_msi[train_mask, :]
adata_msi_test = adata_msi[test_mask, :]

result = method_function(
    adata_rna_train=adata_rna_train,
    adata_rna_test=adata_rna_test,
    adata_msi_train=adata_msi_train,
    adata_msi_test=adata_msi_test,
    params=method_params,
    rna_layer=rna_layer,
    msi_layer=msi_layer,
)

Y_train = result["Y_train"]
Y_train_pred = result["Y_train_pred"]
Y_test = result["Y_test"]
Y_pred = result["Y_pred"]

global_metrics = compute_global_metrics(Y_train, Y_train_pred, Y_test, Y_pred)

global_metrics["task"] = task
global_metrics["method_name"] = method
global_metrics["method_params"] = json.dumps(method_params, sort_keys=True)
global_metrics["hash"] = hash_id
global_metrics["split"] = split_name
global_metrics["rna_layer"] = rna_layer
global_metrics["msi_layer"] = msi_layer
global_metrics["n_train"] = int(train_mask.sum())
global_metrics["n_test"] = int(test_mask.sum())
global_metrics["n_rna_features"] = int(adata_rna.n_vars)
global_metrics["n_msi_targets"] = int(adata_msi.n_vars)

per_metabolite_metrics = compute_per_metabolite_metrics(
    Y_test=Y_test,
    Y_pred=Y_pred,
    metabolite_names=adata_msi_test.var_names,
    task=task,
    method_name=method,
    method_params=json.dumps(method_params, sort_keys=True),
    hash_id=hash_id,
    split=split_name,
)

Path(snakemake.output.global_metrics).parent.mkdir(parents=True, exist_ok=True)
global_metrics.to_csv(snakemake.output.global_metrics, sep="\t", index=False)
per_metabolite_metrics.to_parquet(snakemake.output.per_metabolite_metrics, index=False)

metadata = {
    "task": task,
    "method": method,
    "hash": hash_id,
    "split": split_name,
    "input_rna": str(rna_path),
    "input_metabolomics": str(metab_path),
    "rna_layer": rna_layer,
    "msi_layer": msi_layer,
    "method_params": method_params,
    "n_train": int(train_mask.sum()),
    "n_test": int(test_mask.sum()),
    "n_rna_features": int(adata_rna.n_vars),
    "n_msi_targets": int(adata_msi.n_vars),
    "outputs": {
        "global_metrics": str(snakemake.output.global_metrics),
        "per_metabolite_metrics": str(snakemake.output.per_metabolite_metrics),
        "run_metadata": str(snakemake.output.run_metadata),
    },
}
save_json(snakemake.output.run_metadata, metadata)

# Optional compressed prediction storage. This is intentionally disabled by default
# and only used if the Snakefile/config asks for it and provides this output.
if hasattr(snakemake.output, "predictions_npz"):
    save_predictions_npz(snakemake.output.predictions_npz, Y_test, Y_pred)
