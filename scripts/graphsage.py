"""GraphSAGE node-regression method for the sparkly pipeline.

Same function signature as the other model scripts. torch / torch_geometric
are imported lazily inside ``run_graphsage`` so that importing this module
never breaks the non-GNN methods when torch is not installed.

Graph scope is selected via `graph_scope` (see graph_utils): within_split
(inductive, independent train/test graphs) or transductive (one graph over all
nodes, loss masked to train-fit nodes, early stopping on train-only validation).
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

    transductive = cfg["graph_scope"] == "transductive"

    if transductive:
        full = require_full_graph_kwargs(kwargs, method="graphsage")
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
        aggr=cfg["aggr"],
    )

    if transductive:
        train_result = train_node_regressor_transductive(
            model, prep, cfg, device=kwargs.get("device")
        )
        outputs = assemble_outputs_transductive(prep, train_result)
    else:
        train_result = train_node_regressor(model, prep, cfg, device=kwargs.get("device"))
        outputs = assemble_outputs(prep, train_result)

    outputs["metadata"] = build_metadata(prep, train_result, architecture="graphsage")
    return outputs
