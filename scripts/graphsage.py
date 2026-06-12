"""GraphSAGE node-regression method for the sparkly pipeline.

Same function signature as the other model scripts. torch / torch_geometric
are imported lazily inside ``run_graphsage`` so that importing this module
never breaks the non-GNN methods when torch is not installed.

First-version scope: within-split graph learning only (see graph_utils).
"""

from graph_utils import (
    assemble_outputs,
    build_metadata,
    prepare_within_split,
    resolve_config,
    train_node_regressor,
)


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'graphsage' method requires PyTorch and PyTorch Geometric, but "
            f"they could not be imported ({type(exc).__name__}: {exc}). Install "
            "them in the run environment (e.g. the 'gnn-env' conda env) before "
            "running the gcn / graphsage methods."
        ) from exc


def _build_model(in_dim, hidden_dim, out_dim, num_layers, dropout, aggr):
    import torch
    from torch_geometric.nn import SAGEConv

    class GraphSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = float(dropout)
            self.convs = torch.nn.ModuleList()
            if num_layers <= 1:
                self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggr))
            else:
                self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
                self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggr))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = torch.relu(x)
                    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return x

    return GraphSAGE()


def run_graphsage(
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

    cfg = resolve_config(params, architecture="graphsage")

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
        aggr=cfg["aggr"],
    )

    train_result = train_node_regressor(model, prep, cfg, device=kwargs.get("device"))

    outputs = assemble_outputs(prep, train_result)
    outputs["metadata"] = build_metadata(prep, train_result, architecture="graphsage")
    return outputs
