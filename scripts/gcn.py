"""GCN node-regression method for the sparkly pipeline.

Same function signature as the other model scripts. torch / torch_geometric
are imported lazily inside ``run_gcn`` so that importing this module never
breaks the non-GNN methods when torch is not installed.

First-version scope: within-split graph learning only (see graph_utils). The
train graph is built from the train slice, validation is a held-out subset of
the train slice, and the test graph is built independently from the test
slice. No train-test edges are used.
"""

from graph_utils import (
    assemble_outputs,
    assemble_outputs_transductive,
    build_metadata,
    prepare_transductive,
    prepare_within_split,
    require_full_graph_kwargs,
    resolve_config,
    train_node_regressor,
    train_node_regressor_transductive,
)


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'gcn' method requires PyTorch and PyTorch Geometric, but they "
            f"could not be imported ({type(exc).__name__}: {exc}). Install them "
            "in the run environment (e.g. the 'gnn-env' conda env) before "
            "running the gcn / graphsage methods."
        ) from exc


def _build_model(in_dim, hidden_dim, out_dim, num_layers, dropout):
    import torch
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = float(dropout)
            self.convs = torch.nn.ModuleList()
            if num_layers <= 1:
                self.convs.append(GCNConv(in_dim, out_dim))
            else:
                self.convs.append(GCNConv(in_dim, hidden_dim))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.convs.append(GCNConv(hidden_dim, out_dim))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = torch.relu(x)
                    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return x

    return GCN()


def run_gcn(
    adata_rna_train,
    adata_rna_test,
    adata_msi_train,
    adata_msi_test,
    params,
    rna_layer="X",
    msi_layer="X",
    **kwargs,
):
    _require_torch()

    cfg = resolve_config(params, architecture="gcn")

    transductive = cfg["graph_scope"] == "transductive"

    if transductive:
        full = require_full_graph_kwargs(kwargs, method="gcn")
        prep = prepare_transductive(
            full["adata_rna_full"],
            full["adata_msi_full"],
            full["train_mask"],
            full["test_mask"],
            cfg,
            rna_layer=rna_layer,
            msi_layer=msi_layer,
        )
    else:
        prep = prepare_within_split(
            adata_rna_train,
            adata_rna_test,
            adata_msi_train,
            adata_msi_test,
            cfg,
            rna_layer=rna_layer,
            msi_layer=msi_layer,
        )

    model = _build_model(
        in_dim=prep["n_features"],
        hidden_dim=cfg["hidden_dim"],
        out_dim=prep["n_targets"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

    if transductive:
        train_result = train_node_regressor_transductive(
            model, prep, cfg, device=kwargs.get("device")
        )
        outputs = assemble_outputs_transductive(prep, train_result)
    else:
        train_result = train_node_regressor(model, prep, cfg, device=kwargs.get("device"))
        outputs = assemble_outputs(prep, train_result)

    outputs["metadata"] = build_metadata(prep, train_result, architecture="gcn")
    return outputs
