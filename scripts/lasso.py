from sklearn.linear_model import Lasso

from model_utils import extract_xy


def run_lasso(
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

    model = Lasso(
        alpha=float(params["alpha"]),
        max_iter=int(params.get("max_iter", 10000)),
    )
    model.fit(X_train, Y_train)

    return {
        "Y_train": Y_train,
        "Y_train_pred": model.predict(X_train),
        "Y_test": Y_test,
        "Y_pred": model.predict(X_test),
    }
