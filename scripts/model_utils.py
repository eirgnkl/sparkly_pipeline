import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def to_cpu(x):
    if hasattr(x, "__cuda_array_interface__"):
        import cupy as cp
        return cp.asnumpy(x)
    return np.asarray(x)


def get_matrix(adata, layer_key=None):
    """
    Return matrix from AnnData.

    Defaults to .X. Layer/obsm support is kept so task-level config entries
    such as rna_layer/msi_layer can be used without changing model code.
    """
    if layer_key is None or pd.isna(layer_key):
        return adata.X

    key = str(layer_key).strip()

    if key in {"", "X", "none", "None", "null"}:
        return adata.X

    if key.startswith("layers:"):
        return adata.layers[key.split(":", 1)[1]]

    if key.startswith("obsm:"):
        return adata.obsm[key.split(":", 1)[1]]

    if key in adata.layers:
        return adata.layers[key]

    if key in adata.obsm:
        return adata.obsm[key]

    raise KeyError(
        f"Could not find '{key}' in adata.layers or adata.obsm. "
        f"Available layers={list(adata.layers.keys())}; "
        f"obsm={list(adata.obsm.keys())}"
    )


def prepare_xy(
    adata_rna_train,
    adata_rna_test,
    adata_msi_train,
    adata_msi_test,
    rna_layer="X",
    msi_layer="X",
    dense=True,
):
    X_train = get_matrix(adata_rna_train, rna_layer)
    X_test = get_matrix(adata_rna_test, rna_layer)
    Y_train = get_matrix(adata_msi_train, msi_layer)
    Y_test = get_matrix(adata_msi_test, msi_layer)

    if dense:
        X_train = to_dense(X_train)
        X_test = to_dense(X_test)
        Y_train = to_dense(Y_train)
        Y_test = to_dense(Y_test)

    return X_train, X_test, Y_train, Y_test


def extract_xy(
    adata_rna_train,
    adata_rna_test,
    adata_msi_train,
    adata_msi_test,
    rna_layer="X",
    msi_layer="X",
):
    return prepare_xy(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test,
        rna_layer=rna_layer,
        msi_layer=msi_layer,
        dense=True,
    )


def safe_corr(y_true, y_pred, kind="pearson"):
    y_true = np.asarray(to_cpu(y_true)).ravel()
    y_pred = np.asarray(to_cpu(y_pred)).ravel()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return np.nan

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    if kind == "pearson":
        return pearsonr(y_true, y_pred)[0]
    if kind == "spearman":
        return spearmanr(y_true, y_pred)[0]
    raise ValueError(f"Unknown correlation kind: {kind}")


def average_per_metabolite_corr(Y_true, Y_pred, kind="pearson"):
    Y_true = np.asarray(to_cpu(Y_true))
    Y_pred = np.asarray(to_cpu(Y_pred))
    values = [safe_corr(Y_true[:, j], Y_pred[:, j], kind=kind) for j in range(Y_true.shape[1])]
    return np.nanmean(values)


def average_relative_rmse(Y_true, Y_pred, eps=1e-8):
    Y_true = np.asarray(to_cpu(Y_true))
    Y_pred = np.asarray(to_cpu(Y_pred))
    values = []
    for j in range(Y_true.shape[1]):
        rmse_j = root_mean_squared_error(Y_true[:, j], Y_pred[:, j])
        denom_j = np.mean(np.abs(Y_true[:, j])) + eps
        values.append(rmse_j / denom_j)
    return np.nanmean(values)


def compute_global_metrics(Y_train, Y_train_pred, Y_test, Y_pred):
    """
    One-row global metrics table.

    Keeps backward-compatible old columns (rmse, mae, r2, pearson, spearman),
    while also exposing explicit train/test names for the report.
    """
    Y_train = np.asarray(to_cpu(Y_train))
    Y_train_pred = np.asarray(to_cpu(Y_train_pred))
    Y_test = np.asarray(to_cpu(Y_test))
    Y_pred = np.asarray(to_cpu(Y_pred))

    test_rmse = root_mean_squared_error(Y_test, Y_pred)
    test_mae = mean_absolute_error(Y_test, Y_pred)
    test_r2 = r2_score(Y_test, Y_pred)
    global_test_pearson = safe_corr(Y_test, Y_pred, kind="pearson")
    global_test_spearman = safe_corr(Y_test, Y_pred, kind="spearman")

    train_rmse = root_mean_squared_error(Y_train, Y_train_pred)
    train_mae = mean_absolute_error(Y_train, Y_train_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    global_train_pearson = safe_corr(Y_train, Y_train_pred, kind="pearson")
    global_train_spearman = safe_corr(Y_train, Y_train_pred, kind="spearman")

    avg_pearson_per_metabolite = average_per_metabolite_corr(Y_test, Y_pred, kind="pearson")
    avg_spearman_per_metabolite = average_per_metabolite_corr(Y_test, Y_pred, kind="spearman")
    avg_rel_rmse = average_relative_rmse(Y_test, Y_pred)

    return pd.DataFrame({
        "test_rmse": [test_rmse],
        "test_mae": [test_mae],
        "test_r2": [test_r2],
        "global_test_pearson": [global_test_pearson],
        "global_test_spearman": [global_test_spearman],
        "avg_pearson_per_metabolite": [avg_pearson_per_metabolite],
        "avg_spearman_per_metabolite": [avg_spearman_per_metabolite],
        "avg_rel_rmse": [avg_rel_rmse],
        "train_rmse": [train_rmse],
        "train_mae": [train_mae],
        "train_r2": [train_r2],
        "global_train_pearson": [global_train_pearson],
        "global_train_spearman": [global_train_spearman],    
})


def _parse_mz(name):
    """Extract a numeric m/z value from labels such as 'mz 250.03' when possible."""
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", str(name))
    return float(match.group(0)) if match else np.nan


def compute_per_metabolite_metrics(
    Y_test,
    Y_pred,
    metabolite_names,
    task,
    method_name,
    method_params,
    hash_id,
    split,
    eps=1e-8,
):
    """
    Compact one-row-per-metabolite metrics table.

    This replaces huge prediction TSVs for most use cases. It stores summary
    quality per target, not cell-level predictions.
    """
    Y_test = np.asarray(to_cpu(Y_test))
    Y_pred = np.asarray(to_cpu(Y_pred))

    rows = []
    metabolite_names = pd.Index(metabolite_names).astype(str)

    for j, metabolite_id in enumerate(metabolite_names):
        yt = Y_test[:, j]
        yp = Y_pred[:, j]
        valid = np.isfinite(yt) & np.isfinite(yp)

        if valid.sum() < 1:
            rmse_j = mae_j = r2_j = pearson_j = spearman_j = rel_rmse_j = np.nan
            mean_true = std_true = np.nan
            n_valid = 0
        else:
            yt_valid = yt[valid]
            yp_valid = yp[valid]
            n_valid = int(valid.sum())
            rmse_j = root_mean_squared_error(yt_valid, yp_valid)
            mae_j = mean_absolute_error(yt_valid, yp_valid)
            rel_rmse_j = rmse_j / (np.mean(np.abs(yt_valid)) + eps)
            mean_true = float(np.mean(yt_valid))
            std_true = float(np.std(yt_valid))

            if n_valid >= 2:
                try:
                    r2_j = r2_score(yt_valid, yp_valid)
                except Exception:
                    r2_j = np.nan
            else:
                r2_j = np.nan

            pearson_j = safe_corr(yt_valid, yp_valid, kind="pearson")
            spearman_j = safe_corr(yt_valid, yp_valid, kind="spearman")

        rows.append({
            "task": task,
            "method_name": method_name,
            "method_params": str(method_params),
            "hash": hash_id,
            "split": split,
            "metabolite_index": j,
            "metabolite_id": metabolite_id,
            "mz": _parse_mz(metabolite_id),
            "test_rmse": rmse_j,
            "test_mae": mae_j,
            "test_r2": r2_j,
            "test_pearson": pearson_j,
            "test_spearman": spearman_j,
            "test_rel_rmse": rel_rmse_j,
            "test_true_mean": mean_true,
            "test_true_std": std_true,
            "n_test_valid": n_valid,
        })

    return pd.DataFrame(rows)


def save_predictions_npz(path, Y_test, Y_pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=np.asarray(to_cpu(Y_test), dtype=np.float32),
        y_pred=np.asarray(to_cpu(Y_pred), dtype=np.float32),
    )


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
