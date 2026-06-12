"""Graph construction and QC helpers for the GCN / GraphSAGE methods.

This module is intentionally framework-agnostic: it depends only on numpy,
scipy and scikit-learn, all of which are already in the pipeline environment.
It does NOT import torch or torch_geometric, so importing it never breaks the
non-GNN methods. Torch is only needed inside scripts/gcn.py and
scripts/graphsage.py.

Two graph sources are supported (and only these two):

* ``obsp``              - reuse a precomputed adjacency in ``adata.obsp[key]``
* ``radius_capped_knn`` - build a kNN graph from ``adata.obsm['spatial']`` and
                          cap edge length by an adaptive radius.

All graphs are treated as undirected and stored as a unique edge list with
``i < j`` (shape ``(n_edges, 2)``). Observation order is always preserved:
node ``k`` corresponds to row ``k`` of the AnnData slice it was built from.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from model_utils import get_matrix, to_dense


DEFAULT_OBSP_KEY = "spatial_connectivities"
SUPPORTED_GRAPH_SOURCES = {"obsp", "radius_capped_knn"}
SUPPORTED_RADIUS_STRATEGIES = {"kth_neighbor_percentile"}


def _to_py(value):
    """Cast numpy scalars to plain python so metadata is JSON-friendly."""
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        value = float(value)
    elif isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _unique_undirected(edges):
    """Return unique undirected edges as an ``(E, 2)`` int array with i < j."""
    if edges is None or len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.asarray(edges, dtype=np.int64)
    # Drop self loops and orient each edge so that the smaller index is first.
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)

    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    oriented = np.stack([lo, hi], axis=1)
    return np.unique(oriented, axis=0)


def _edges_from_obsp(adata, obsp_key):
    if obsp_key not in adata.obsp:
        raise KeyError(
            f"graph_source='obsp' requested but obsp key '{obsp_key}' was not "
            f"found. Available obsp keys: {list(adata.obsp.keys())}. "
            "Refusing to silently fall back to another graph."
        )

    A = adata.obsp[obsp_key]
    A = coo_matrix(A)
    pairs = np.stack([A.row, A.col], axis=1)
    return _unique_undirected(pairs)


def _get_coords(adata):
    if "spatial" not in adata.obsm:
        raise KeyError(
            "graph_source='radius_capped_knn' requires spatial coordinates in "
            f"adata.obsm['spatial'], but available obsm keys are: "
            f"{list(adata.obsm.keys())}."
        )
    return np.asarray(adata.obsm["spatial"], dtype=float)


def _radius_capped_knn(
    coords,
    knn_k=6,
    radius_strategy="kth_neighbor_percentile",
    radius_percentile=90,
    max_radius_multiplier=5,
    repair_isolates=1,
):
    """Build a symmetric kNN graph capped by an adaptive radius.

    Returns ``(edges, meta)`` where ``edges`` is the unique undirected edge
    list and ``meta`` records the chosen radius and repair statistics. The
    procedure is fully unsupervised: it uses only coordinates.
    """
    if radius_strategy not in SUPPORTED_RADIUS_STRATEGIES:
        raise ValueError(
            f"radius_strategy='{radius_strategy}' is not implemented. "
            f"Supported strategies: {sorted(SUPPORTED_RADIUS_STRATEGIES)}."
        )

    n = coords.shape[0]
    knn_k = int(knn_k)
    if knn_k < 1:
        raise ValueError(f"knn_k must be >= 1, got {knn_k}.")

    # Query knn_k neighbours plus the node itself.
    k_query = min(knn_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_query)
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)

    # Column 0 is the node itself (distance 0); columns 1..knn_k are neighbours.
    knn_eff = max(k_query - 1, 0)
    if knn_eff == 0:
        # Degenerate single-node case.
        empty = np.empty((0, 2), dtype=np.int64)
        meta = {
            "chosen_radius": None,
            "radius_from_percentile": None,
            "radius_cap": None,
            "n_edges_removed_by_radius": 0,
            "n_isolates_repaired": 0,
        }
        return empty, meta

    nearest_neighbor_dist = dist[:, 1]
    kth_neighbor_dist = dist[:, knn_eff]  # knn_k-th neighbour (or last available)

    # Directed kNN edges with their Euclidean lengths.
    src = np.repeat(np.arange(n), knn_eff)
    dst = idx[:, 1 : 1 + knn_eff].reshape(-1)
    directed = np.stack([src, dst], axis=1)
    edges = _unique_undirected(directed)

    n_edges_before = len(edges)

    # Adaptive radius (unsupervised).
    radius_from_percentile = float(np.percentile(kth_neighbor_dist, radius_percentile))
    radius_cap = float(max_radius_multiplier) * float(np.median(nearest_neighbor_dist))
    chosen_radius = float(min(radius_from_percentile, radius_cap))

    # Remove edges longer than the chosen radius.
    if len(edges) > 0:
        lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        edges = edges[lengths <= chosen_radius]
    n_edges_removed_by_radius = int(n_edges_before - len(edges))

    # Reconnect isolated nodes to their nearest neighbour, if requested.
    n_isolates_repaired = 0
    if int(repair_isolates) == 1 and n > 1:
        deg = np.bincount(edges.reshape(-1), minlength=n) if len(edges) else np.zeros(n, dtype=int)
        isolates = np.where(deg == 0)[0]
        repair_pairs = []
        for node in isolates:
            partner = int(idx[node, 1])  # nearest neighbour excluding self
            if partner == node:
                continue
            repair_pairs.append((node, partner))
        if repair_pairs:
            combined = np.vstack([edges, np.asarray(repair_pairs, dtype=np.int64)])
            edges_after = _unique_undirected(combined)
            n_isolates_repaired = int(len(edges_after) - len(edges))
            edges = edges_after

    meta = {
        "chosen_radius": chosen_radius,
        "radius_from_percentile": radius_from_percentile,
        "radius_cap": radius_cap,
        "n_edges_removed_by_radius": n_edges_removed_by_radius,
        "n_isolates_repaired": n_isolates_repaired,
        # Repaired edges may exceed chosen_radius by construction.
        "repaired_edges_may_exceed_radius": bool(n_isolates_repaired > 0),
    }
    return edges, meta


def compute_graph_qc(
    n_nodes,
    edges,
    coords=None,
    knn_k=None,
    chosen_radius=None,
    n_edges_removed_by_radius=None,
    n_isolates_repaired=None,
):
    """Compute unsupervised quality-control statistics for one graph.

    ``edges`` is a unique undirected edge list (``(E, 2)``). No target values
    or model scores are used here.
    """
    edges = np.asarray(edges, dtype=np.int64) if edges is not None else np.empty((0, 2), np.int64)
    n_nodes = int(n_nodes)
    n_edges = int(len(edges))

    if n_nodes > 0:
        if n_edges > 0:
            degrees = np.bincount(edges.reshape(-1), minlength=n_nodes)
        else:
            degrees = np.zeros(n_nodes, dtype=int)
    else:
        degrees = np.zeros(0, dtype=int)

    isolated = int((degrees == 0).sum()) if n_nodes > 0 else 0
    isolated_fraction = float(isolated / n_nodes) if n_nodes > 0 else None

    # Connected components on a symmetric adjacency.
    if n_nodes > 0 and n_edges > 0:
        data = np.ones(n_edges, dtype=np.int8)
        adj = csr_matrix(
            (np.concatenate([data, data]),
             (np.concatenate([edges[:, 0], edges[:, 1]]),
              np.concatenate([edges[:, 1], edges[:, 0]]))),
            shape=(n_nodes, n_nodes),
        )
        n_components, labels = connected_components(adj, directed=False)
        comp_sizes = np.bincount(labels)
        giant_fraction = float(comp_sizes.max() / n_nodes)
    else:
        n_components = n_nodes
        giant_fraction = float(1.0 / n_nodes) if n_nodes > 0 else None

    median_len = p95_len = p99_len = None
    if coords is not None and n_edges > 0:
        coords = np.asarray(coords, dtype=float)
        lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        median_len = float(np.median(lengths))
        p95_len = float(np.percentile(lengths, 95))
        p99_len = float(np.percentile(lengths, 99))

    qc = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": _to_py(float(degrees.mean())) if n_nodes > 0 else None,
        "median_degree": _to_py(float(np.median(degrees))) if n_nodes > 0 else None,
        "max_degree": _to_py(int(degrees.max())) if n_nodes > 0 else None,
        "isolated_nodes": isolated,
        "isolated_fraction": _to_py(isolated_fraction),
        "n_components": int(n_components),
        "giant_component_fraction": _to_py(giant_fraction),
        "median_edge_length": _to_py(median_len),
        "p95_edge_length": _to_py(p95_len),
        "p99_edge_length": _to_py(p99_len),
        "chosen_radius": _to_py(chosen_radius),
        "knn_k": _to_py(knn_k),
        "n_edges_removed_by_radius": _to_py(n_edges_removed_by_radius),
        "n_isolates_repaired": _to_py(n_isolates_repaired),
    }
    return qc


def build_graph(
    adata,
    graph_source="obsp",
    obsp_key=DEFAULT_OBSP_KEY,
    knn_k=6,
    radius_strategy="kth_neighbor_percentile",
    radius_percentile=90,
    max_radius_multiplier=5,
    repair_isolates=1,
):
    """Build one undirected graph for an (already-sliced) AnnData object.

    Returns a dict with the unique undirected ``edges`` (``(E, 2)``), the node
    ``coords`` (or ``None``), the construction ``meta`` and the ``qc`` block.
    Observation order is preserved: node ``k`` == row ``k`` of ``adata``.
    """
    if graph_source not in SUPPORTED_GRAPH_SOURCES:
        raise ValueError(
            f"graph_source='{graph_source}' is not supported. "
            f"Supported sources: {sorted(SUPPORTED_GRAPH_SOURCES)}."
        )

    n_nodes = int(adata.n_obs)
    coords = np.asarray(adata.obsm["spatial"], dtype=float) if "spatial" in adata.obsm else None

    if graph_source == "obsp":
        edges = _edges_from_obsp(adata, obsp_key)
        meta = {
            "chosen_radius": None,
            "n_edges_removed_by_radius": 0,
            "n_isolates_repaired": 0,
        }
    else:  # radius_capped_knn
        coords = _get_coords(adata)
        edges, meta = _radius_capped_knn(
            coords,
            knn_k=knn_k,
            radius_strategy=radius_strategy,
            radius_percentile=radius_percentile,
            max_radius_multiplier=max_radius_multiplier,
            repair_isolates=repair_isolates,
        )

    qc = compute_graph_qc(
        n_nodes,
        edges,
        coords=coords,
        knn_k=knn_k,
        chosen_radius=meta.get("chosen_radius"),
        n_edges_removed_by_radius=meta.get("n_edges_removed_by_radius"),
        n_isolates_repaired=meta.get("n_isolates_repaired"),
    )

    return {
        "n_nodes": n_nodes,
        "edges": edges,
        "coords": coords,
        "meta": meta,
        "qc": qc,
    }


def induce_subgraph(edges, node_positions):
    """Induce a subgraph on ``node_positions`` and relabel to ``0..k-1``.

    ``node_positions`` is an array of original node indices to keep. The
    returned edges are relabelled so that local node ``k`` corresponds to
    ``node_positions[k]`` (i.e. the order of ``node_positions`` is preserved).
    """
    node_positions = np.asarray(node_positions, dtype=np.int64)
    if edges is None or len(edges) == 0 or len(node_positions) == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.asarray(edges, dtype=np.int64)

    # Map original index -> local index; -1 if not kept.
    max_orig = max(int(edges.max()), int(node_positions.max())) + 1
    remap = np.full(max_orig, -1, dtype=np.int64)
    remap[node_positions] = np.arange(len(node_positions), dtype=np.int64)

    a = remap[edges[:, 0]]
    b = remap[edges[:, 1]]
    mask = (a >= 0) & (b >= 0)
    sub = np.stack([a[mask], b[mask]], axis=1)
    return _unique_undirected(sub)


# ---------------------------------------------------------------------------
# Parameter parsing + within-split preparation
# ---------------------------------------------------------------------------

TRANSDUCTIVE_ERROR = (
    "graph_scope='transductive' is not implemented in the current first-version "
    "GCN/GraphSAGE integration. Use graph_scope='within_split'."
)


def _as_str(params, key, default):
    value = params.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _as_int(params, key, default):
    value = params.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return int(default)
    return int(float(value))


def _as_float(params, key, default):
    value = params.get(key, default)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return float(default)
    return float(value)


def _as_bool(params, key, default):
    value = params.get(key, default)
    if isinstance(value, str):
        value = value.strip()
    try:
        return bool(int(float(value)))
    except (TypeError, ValueError):
        return bool(value)


def resolve_config(params, architecture):
    """Resolve raw param strings into a typed config with defaults."""
    graph_scope = _as_str(params, "graph_scope", "within_split")
    if graph_scope == "transductive":
        raise NotImplementedError(TRANSDUCTIVE_ERROR)
    if graph_scope != "within_split":
        raise ValueError(
            f"graph_scope='{graph_scope}' is not supported. "
            "Only 'within_split' is implemented in this first version."
        )

    cfg = {
        "architecture": architecture,
        "graph_source": _as_str(params, "graph_source", "obsp"),
        "obsp_key": _as_str(params, "obsp_key", DEFAULT_OBSP_KEY),
        "knn_k": _as_int(params, "knn_k", 6),
        "radius_strategy": _as_str(params, "radius_strategy", "kth_neighbor_percentile"),
        "radius_percentile": _as_float(params, "radius_percentile", 90),
        "max_radius_multiplier": _as_float(params, "max_radius_multiplier", 5),
        "repair_isolates": int(_as_bool(params, "repair_isolates", 1)),
        "graph_scope": graph_scope,
        "hidden_dim": _as_int(params, "hidden_dim", 64),
        "num_layers": _as_int(params, "num_layers", 2),
        "dropout": _as_float(params, "dropout", 0.2),
        "lr": _as_float(params, "lr", 0.001),
        "weight_decay": _as_float(params, "weight_decay", 0.0001),
        "epochs": _as_int(params, "epochs", 300),
        "patience": _as_int(params, "patience", 30),
        "val_fraction": _as_float(params, "val_fraction", 0.15),
        "val_strategy": _as_str(params, "val_strategy", "spatial_band"),
        "val_axis": _as_str(params, "val_axis", "auto"),
        "standardize": int(_as_bool(params, "standardize", 1)),
        "seed": _as_int(params, "seed", 666),
        "aggr": _as_str(params, "aggr", "mean"),
    }
    return cfg


def _subgraph_coords(coords, positions):
    if coords is None:
        return None
    return coords[np.asarray(positions, dtype=np.int64)]


SUPPORTED_VAL_STRATEGIES = {"spatial_band", "random"}


def _resolve_val_axis(train_coords, val_axis):
    """Resolve the spatial axis used to carve the validation band.

    'auto' picks the axis with the largest spatial extent (max - min). Explicit
    values 0/1 or 'x'/'y' are also accepted.
    """
    n_dims = train_coords.shape[1]
    val_axis = str(val_axis).strip().lower()

    if val_axis in {"auto", "", "none"}:
        extents = train_coords.max(axis=0) - train_coords.min(axis=0)
        return int(np.argmax(extents))

    alias = {"x": 0, "y": 1, "z": 2}
    if val_axis in alias:
        axis = alias[val_axis]
    else:
        try:
            axis = int(float(val_axis))
        except (TypeError, ValueError):
            raise ValueError(
                f"val_axis='{val_axis}' is invalid. Use 'auto', an integer axis "
                "index, or one of 'x'/'y'/'z'."
            )

    if axis < 0 or axis >= n_dims:
        raise ValueError(
            f"val_axis resolved to {axis}, but coordinates only have {n_dims} "
            "dimensions."
        )
    return axis


def _select_validation(train_coords, n_train, n_val, cfg):
    """Return (fit_idx, val_idx) for the configured validation strategy.

    The validation set is always drawn ONLY from train nodes. 'spatial_band'
    carves a contiguous coordinate slab (the top val_fraction along the chosen
    axis) so the induced validation subgraph stays connected; 'random' keeps the
    original behaviour for comparison.
    """
    strategy = cfg["val_strategy"]

    if strategy == "random":
        perm = np.random.RandomState(cfg["seed"]).permutation(n_train)
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    if strategy == "spatial_band":
        if train_coords is None:
            raise ValueError(
                "val_strategy='spatial_band' requires spatial coordinates in "
                "adata.obsm['spatial'], but none were available for the train "
                "slice. Use val_strategy='random' or provide coordinates."
            )
        axis = _resolve_val_axis(train_coords, cfg["val_axis"])
        order = np.argsort(train_coords[:, axis], kind="stable")  # ascending
        val_idx = np.sort(order[-n_val:])  # top val_fraction slab (high end)
        fit_idx = np.sort(order[:-n_val])
        return fit_idx, val_idx

    raise ValueError(
        f"val_strategy='{strategy}' is not supported. "
        f"Use one of {sorted(SUPPORTED_VAL_STRATEGIES)}."
    )


def prepare_within_split(
    adata_rna_train,
    adata_rna_test,
    adata_msi_train,
    adata_msi_test,
    cfg,
    rna_layer="X",
    msi_layer="X",
):
    """Build the within-split graphs, features, targets and QC.

    Train graph is built from the train slice; the validation graph is the
    subgraph induced on a held-out subset of train nodes; the test graph is
    built independently from the test slice. No train-test edges exist and the
    validation set is drawn ONLY from the train slice.
    """
    # Dense feature/target matrices, reusing the existing extraction utility.
    X_train_full = np.asarray(to_dense(get_matrix(adata_rna_train, rna_layer)), dtype=float)
    X_test_raw = np.asarray(to_dense(get_matrix(adata_rna_test, rna_layer)), dtype=float)
    Y_train_full = np.asarray(to_dense(get_matrix(adata_msi_train, msi_layer)), dtype=float)
    Y_test = np.asarray(to_dense(get_matrix(adata_msi_test, msi_layer)), dtype=float)

    if X_train_full.ndim == 1:
        X_train_full = X_train_full.reshape(-1, 1)
    if X_test_raw.ndim == 1:
        X_test_raw = X_test_raw.reshape(-1, 1)
    if Y_train_full.ndim == 1:
        Y_train_full = Y_train_full.reshape(-1, 1)
    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)

    n_train = X_train_full.shape[0]
    n_targets = Y_train_full.shape[1]

    # Build full train graph and test graph (independent, within-split).
    train_graph = build_graph(
        adata_rna_train,
        graph_source=cfg["graph_source"],
        obsp_key=cfg["obsp_key"],
        knn_k=cfg["knn_k"],
        radius_strategy=cfg["radius_strategy"],
        radius_percentile=cfg["radius_percentile"],
        max_radius_multiplier=cfg["max_radius_multiplier"],
        repair_isolates=cfg["repair_isolates"],
    )
    test_graph = build_graph(
        adata_rna_test,
        graph_source=cfg["graph_source"],
        obsp_key=cfg["obsp_key"],
        knn_k=cfg["knn_k"],
        radius_strategy=cfg["radius_strategy"],
        radius_percentile=cfg["radius_percentile"],
        max_radius_multiplier=cfg["max_radius_multiplier"],
        repair_isolates=cfg["repair_isolates"],
    )

    # Validation split: drawn ONLY from train nodes.
    val_fraction = cfg["val_fraction"]
    n_val = int(round(val_fraction * n_train))
    if n_val < 1:
        raise ValueError(
            f"val_fraction={val_fraction} yields 0 validation nodes for "
            f"n_train={n_train}. Increase val_fraction or the train set size."
        )
    if n_train - n_val < 1:
        raise ValueError(
            f"val_fraction={val_fraction} leaves 0 train-fit nodes for "
            f"n_train={n_train}. Decrease val_fraction."
        )

    train_coords = train_graph["coords"]
    fit_idx, val_idx = _select_validation(train_coords, n_train, n_val, cfg)

    if cfg["val_strategy"] == "spatial_band" and train_coords is not None:
        val_axis_resolved = _resolve_val_axis(train_coords, cfg["val_axis"])
    else:
        val_axis_resolved = None

    # Feature standardization: fit ONLY on train-fit nodes (never on test).
    if cfg["standardize"]:
        scaler = StandardScaler()
        scaler.fit(X_train_full[fit_idx])
        X_fit = scaler.transform(X_train_full[fit_idx])
        X_val = scaler.transform(X_train_full[val_idx])
        X_test = scaler.transform(X_test_raw)
    else:
        X_fit = X_train_full[fit_idx]
        X_val = X_train_full[val_idx]
        X_test = X_test_raw

    Y_fit = Y_train_full[fit_idx]
    Y_val = Y_train_full[val_idx]

    # Induced subgraphs (relabelled to local 0..k-1, preserving order).
    edges_fit = induce_subgraph(train_graph["edges"], fit_idx)
    edges_val = induce_subgraph(train_graph["edges"], val_idx)
    edges_test = test_graph["edges"]

    chosen_radius_train = train_graph["meta"].get("chosen_radius")

    qc = {
        "train": train_graph["qc"],
        "train_fit": compute_graph_qc(
            len(fit_idx), edges_fit,
            coords=_subgraph_coords(train_coords, fit_idx),
            knn_k=cfg["knn_k"], chosen_radius=chosen_radius_train,
        ),
        "val": compute_graph_qc(
            len(val_idx), edges_val,
            coords=_subgraph_coords(train_coords, val_idx),
            knn_k=cfg["knn_k"], chosen_radius=chosen_radius_train,
        ),
        "test": test_graph["qc"],
    }

    return {
        "X_fit": X_fit, "X_val": X_val, "X_test": X_test,
        "Y_fit": Y_fit, "Y_val": Y_val, "Y_test": Y_test,
        "Y_train_full": Y_train_full,
        "edges_fit": edges_fit, "edges_val": edges_val, "edges_test": edges_test,
        "fit_idx": fit_idx, "val_idx": val_idx,
        "n_train": n_train, "n_features": X_train_full.shape[1], "n_targets": n_targets,
        "qc": qc,
        "train_meta": train_graph["meta"], "test_meta": test_graph["meta"],
        "val_axis_resolved": val_axis_resolved,
        "cfg": cfg,
    }


def train_node_regressor(model, prep, cfg, device=None):
    """Train a node-regression model with validation-only early stopping.

    Torch is imported lazily here so the module stays importable without torch.
    Early stopping monitors the validation loss only; test data is never used
    for validation, early stopping or model selection.
    """
    import torch

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = model.to(dev)

    def to_edge_index(edges):
        if edges is None or len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=dev)
        e = np.asarray(edges, dtype=np.int64).T  # (2, E)
        bidir = np.concatenate([e, e[::-1]], axis=1)  # undirected
        return torch.tensor(bidir, dtype=torch.long, device=dev)

    x_fit = torch.tensor(prep["X_fit"], dtype=torch.float32, device=dev)
    y_fit = torch.tensor(prep["Y_fit"], dtype=torch.float32, device=dev)
    ei_fit = to_edge_index(prep["edges_fit"])

    x_val = torch.tensor(prep["X_val"], dtype=torch.float32, device=dev)
    y_val = torch.tensor(prep["Y_val"], dtype=torch.float32, device=dev)
    ei_val = to_edge_index(prep["edges_val"])

    x_test = torch.tensor(prep["X_test"], dtype=torch.float32, device=dev)
    ei_test = to_edge_index(prep["edges_test"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0

    for epoch in range(int(cfg["epochs"])):
        model.train()
        optimizer.zero_grad()
        out = model(x_fit, ei_fit)
        loss = loss_fn(out, y_fit)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val, ei_val), y_val).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(cfg["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        fit_pred = model(x_fit, ei_fit).cpu().numpy()
        val_pred = model(x_val, ei_val).cpu().numpy()
        test_pred = model(x_test, ei_test).cpu().numpy()

    return {
        "fit_pred": fit_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "device": str(dev),
    }


def assemble_outputs(prep, train_result):
    """Combine fit + val predictions back into original train-slice order."""
    n_train = prep["n_train"]
    n_targets = prep["n_targets"]

    Y_train_pred = np.empty((n_train, n_targets), dtype=float)
    Y_train_pred[prep["fit_idx"]] = train_result["fit_pred"]
    Y_train_pred[prep["val_idx"]] = train_result["val_pred"]

    return {
        "Y_train": prep["Y_train_full"],
        "Y_train_pred": Y_train_pred,
        "Y_test": prep["Y_test"],
        "Y_pred": train_result["test_pred"],
    }


def build_metadata(prep, train_result, architecture):
    """Assemble the metadata block returned to run_method.py."""
    cfg = prep["cfg"]
    metadata = {
        "architecture": architecture,
        "graph_source": cfg["graph_source"],
        "graph_scope": cfg["graph_scope"],
        "obsp_key": cfg["obsp_key"],
        "knn_k": cfg["knn_k"],
        "radius_strategy": cfg["radius_strategy"],
        "radius_percentile": cfg["radius_percentile"],
        "max_radius_multiplier": cfg["max_radius_multiplier"],
        "repair_isolates": cfg["repair_isolates"],
        "chosen_radius": _to_py(prep["train_meta"].get("chosen_radius")),
        "chosen_radius_test": _to_py(prep["test_meta"].get("chosen_radius")),
        "val_fraction": cfg["val_fraction"],
        "val_strategy": cfg["val_strategy"],
        "val_axis": cfg["val_axis"],
        "val_axis_resolved": _to_py(prep.get("val_axis_resolved")),
        "standardize": cfg["standardize"],
        "seed": cfg["seed"],
        "n_train_nodes": int(prep["n_train"]),
        "n_train_fit_nodes": int(len(prep["fit_idx"])),
        "n_val_nodes": int(len(prep["val_idx"])),
        "n_test_nodes": int(prep["Y_test"].shape[0]),
        "graph_qc": prep["qc"],
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "device": train_result["device"],
    }
    if architecture == "graphsage":
        metadata["aggr"] = cfg["aggr"]
    return metadata
