import numpy as np
import pandas as pd
import cupy as cp
from xgboost import XGBRegressor

from model_utils import extract_xy


def ensure_gpu(data):
    if hasattr(data, "__cuda_array_interface__"):
        return data
    if isinstance(data, pd.DataFrame):
        return cp.asarray(data.values)
    return cp.asarray(data)


def ensure_cpu(data):
    if hasattr(data, "__cuda_array_interface__"):
        return cp.asnumpy(data)
    return np.asarray(data)


def run_xgboost(
    adata_rna_train,
    adata_rna_test,
    adata_msi_train,
    adata_msi_test,
    params,
    rna_layer="X",
    msi_layer="X",
    **kwargs,
):
    X_train, X_test, Y_train, Y_test = extract_xy(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test,
        rna_layer=rna_layer,
        msi_layer=msi_layer,
    )

    X_train_gpu = ensure_gpu(X_train)
    X_test_gpu = ensure_gpu(X_test)
    Y_train_gpu = ensure_gpu(Y_train)

    model = XGBRegressor(
        device="cuda",
        tree_method="hist",
        objective="reg:squarederror",
        reg_alpha=float(params.get("alpha", params.get("reg_alpha", 10))),
        reg_lambda=float(params.get("lambda", params.get("reg_lambda", 50))),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        n_estimators=int(params.get("n_estimators", 500)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.7)),
        min_child_weight=float(params.get("min_child_weight", 2)),
        n_jobs=int(params.get("n_jobs", 15)),
        random_state=int(params.get("seed", params.get("random_state", 666))),
    )

    model.fit(X_train_gpu, Y_train_gpu, verbose=False)

    return {
        "Y_train": ensure_cpu(Y_train_gpu),
        "Y_train_pred": ensure_cpu(model.predict(X_train_gpu)),
        "Y_test": np.asarray(Y_test),
        "Y_pred": ensure_cpu(model.predict(X_test_gpu)),
    }
