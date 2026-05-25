from pathlib import Path
import html

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


TOP_N = 5

PRIMARY_SELECTION_METRICS = {
    "test_r2": "max",
    "avg_pearson_per_metabolite": "max",
    "avg_rel_rmse": "min",
}

METRIC_LABELS = {
    # Primary global-summary metrics
    "test_r2": "Mean per-metabolite R²",
    "avg_pearson_per_metabolite": "Mean per-metabolite Pearson",
    "avg_spearman_per_metabolite": "Mean per-metabolite Spearman",
    "avg_rel_rmse": "Mean per-metabolite relative RMSE",

    # Flattened/global correlations in global_metrics.tsv
    "global_test_pearson": "Global Pearson",
    "global_test_spearman": "Global Spearman",
    "global_train_pearson": "Global train Pearson",
    "global_train_spearman": "Global train Spearman",

    # Error metrics
    "test_rmse": "RMSE",
    "test_mae": "MAE",
    "train_rmse": "Train RMSE",
    "train_mae": "Train MAE",
    "train_r2": "Train R²",

    # Per-metabolite metrics
    "test_rel_rmse": "Relative RMSE",
    "test_pearson": "Per-metabolite Pearson",
    "test_spearman": "Per-metabolite Spearman",
    "test_true_mean": "Test true mean",
    "test_true_std": "Test true std",
    "test_mean_true": "Test true mean",
    "test_std_true": "Test true std",
    "mean_true": "Test true mean",
    "std_true": "Test true std",

    # Metadata / ranking columns
    "selection_metric_label": "Selected by",
    "selection_rank": "Rank",
    "method_name": "Model",
    "hash": "Run hash",
    "split": "Split",
    "method_params": "Parameters",
    "metabolite_index": "Metabolite index",
    "metabolite_id": "Metabolite ID",
    "mz": "m/z",
    "n_test_valid": "Valid test spots",
    "rank_test_r2": "Rank: R²",
    "rank_avg_pearson_per_metabolite": "Rank: per-met Pearson",
    "rank_avg_rel_rmse": "Rank: rel RMSE",
    "mean_primary_rank": "Mean primary rank",
    "selected_by_label": "Selected by",
}

RUN_TABLE_COLUMNS = [
    "selection_metric_label",
    "selection_rank",
    "method_name",
    "hash",
    "split",
    "test_r2",
    "avg_pearson_per_metabolite",
    "avg_rel_rmse",
    "global_test_pearson",
    "global_test_spearman",
    "test_rmse",
    "test_mae",
    "method_params",  # keep params at far right
]

PER_MODEL_TABLE_COLUMNS = [
    "method_name",
    "hash",
    "split",
    "test_r2",
    "avg_pearson_per_metabolite",
    "avg_rel_rmse",
    "global_test_pearson",
    "global_test_spearman",
    "test_rmse",
    "test_mae",
    "method_params",  # keep params at far right
]

RANKING_TABLE_COLUMNS = [
    "method_name",
    "hash",
    "split",
    "rank_test_r2",
    "rank_avg_pearson_per_metabolite",
    "rank_avg_rel_rmse",
    "mean_primary_rank",
    "test_r2",
    "avg_pearson_per_metabolite",
    "avg_rel_rmse",
    "global_test_pearson",
    "method_params",  # keep params at far right
]

METABOLITE_TABLE_COLUMNS = [
    "metabolite_index",
    "metabolite_id",
    "mz",
    "test_r2",
    "test_pearson",
    "test_spearman",
    "test_rel_rmse",
    "test_rmse",
    "test_mae",
    "test_true_mean",
    "test_true_std",
    "n_test_valid",
]


def _fmt_float(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def _fmt_params(value, max_len=220):
    if pd.isna(value):
        return ""
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _label(col):
    return METRIC_LABELS.get(col, col)


def _display_table(df, columns=None, max_rows=20):
    if df is None or df.empty:
        return "<p>No data available.</p>"

    show = df.copy()

    if columns is not None:
        columns = [c for c in columns if c in show.columns]
        show = show[columns]

    show = show.head(max_rows).copy()

    # Keep parameters visible, but compact and right-most according to column lists.
    if "method_params" in show.columns:
        show["method_params"] = show["method_params"].map(_fmt_params)

    numeric_cols = show.select_dtypes(include=["float", "float64", "float32", "int", "int64", "int32"]).columns
    for col in numeric_cols:
        show[col] = show[col].map(_fmt_float)

    show = show.rename(columns={c: _label(c) for c in show.columns})

    return show.to_html(index=False, escape=True, classes="report-table")


def _fig_html(fig):
    if fig is None:
        return "<p>Plot could not be generated.</p>"
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _safe_col(df, col):
    return col in df.columns and df[col].notna().any()


def _sort_by_metric(df, metric, direction):
    valid = df[df[metric].notna()].copy()
    return valid.sort_values(metric, ascending=(direction == "min"))


def _make_top_runs(global_df, top_n=TOP_N):
    rows = []
    for metric, direction in PRIMARY_SELECTION_METRICS.items():
        if metric not in global_df.columns:
            continue

        ranked = _sort_by_metric(global_df, metric, direction).head(top_n).copy()
        ranked["selection_metric"] = metric
        ranked["selection_metric_label"] = _label(metric)
        ranked["selection_direction"] = direction
        ranked["selection_rank"] = range(1, len(ranked) + 1)
        rows.append(ranked)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _make_best_per_model_by_r2(global_df):
    if "test_r2" not in global_df.columns or "method_name" not in global_df.columns:
        return pd.DataFrame()

    ranked = global_df[global_df["test_r2"].notna()].sort_values("test_r2", ascending=False)

    if ranked.empty:
        return pd.DataFrame()

    return ranked.groupby("method_name", as_index=False, sort=False).head(1).copy()


def _make_ranking_comparison(global_df, top_runs):
    if top_runs.empty:
        return pd.DataFrame()

    rank_df = global_df.copy()

    for metric, direction in PRIMARY_SELECTION_METRICS.items():
        if metric not in rank_df.columns:
            continue

        rank_df[f"rank_{metric}"] = rank_df[metric].rank(
            ascending=(direction == "min"),
            method="min",
        )

    rank_cols = [f"rank_{m}" for m in PRIMARY_SELECTION_METRICS if f"rank_{m}" in rank_df.columns]
    if rank_cols:
        rank_df["mean_primary_rank"] = rank_df[rank_cols].mean(axis=1)

    key_cols = ["method_name", "hash", "split"]
    existing_key_cols = [c for c in key_cols if c in top_runs.columns and c in rank_df.columns]

    selected_keys = top_runs[existing_key_cols].drop_duplicates()
    out = selected_keys.merge(rank_df, on=existing_key_cols, how="left")

    if "mean_primary_rank" in out.columns:
        out = out.sort_values("mean_primary_rank", ascending=True)

    return out


def _selected_unique_runs(top_runs):
    if top_runs.empty:
        return pd.DataFrame()

    best_rows = top_runs[top_runs["selection_rank"] == 1].copy()
    if best_rows.empty:
        return pd.DataFrame()

    key_cols = ["task", "method_name", "hash", "split"]
    key_cols = [c for c in key_cols if c in best_rows.columns]

    selected = []
    for _, group in best_rows.groupby(key_cols, dropna=False):
        row = group.iloc[0].copy()
        labels = group["selection_metric_label"].dropna().astype(str).tolist()
        row["selected_by_label"] = ", ".join(labels)
        selected.append(row)

    return pd.DataFrame(selected)


def _add_compatibility_aliases(global_df, per_met_df):
    """
    Keep report compatible with both:
    - old global_metrics.tsv: test_pearson/test_spearman are flattened/global
    - new global_metrics.tsv: global_test_pearson/global_test_spearman
    Also normalizes test true mean/std aliases in per-metabolite files.
    """
    global_df = global_df.copy()
    per_met_df = per_met_df.copy()

    if "global_test_pearson" not in global_df.columns and "test_pearson" in global_df.columns:
        global_df["global_test_pearson"] = global_df["test_pearson"]

    if "global_test_spearman" not in global_df.columns and "test_spearman" in global_df.columns:
        global_df["global_test_spearman"] = global_df["test_spearman"]

    if "global_train_pearson" not in global_df.columns and "train_pearson" in global_df.columns:
        global_df["global_train_pearson"] = global_df["train_pearson"]

    if "global_train_spearman" not in global_df.columns and "train_spearman" in global_df.columns:
        global_df["global_train_spearman"] = global_df["train_spearman"]

    # Your current model_utils writes test_true_mean/test_true_std.
    if "test_true_mean" not in per_met_df.columns:
        if "test_mean_true" in per_met_df.columns:
            per_met_df["test_true_mean"] = per_met_df["test_mean_true"]
        elif "mean_true" in per_met_df.columns:
            per_met_df["test_true_mean"] = per_met_df["mean_true"]

    if "test_true_std" not in per_met_df.columns:
        if "test_std_true" in per_met_df.columns:
            per_met_df["test_true_std"] = per_met_df["test_std_true"]
        elif "std_true" in per_met_df.columns:
            per_met_df["test_true_std"] = per_met_df["std_true"]

    return global_df, per_met_df


def _get_snakemake_io():
    """Support both named and positional Snakemake IO fields."""
    global_path = getattr(snakemake.input, "global_metrics", snakemake.input[0])
    per_met_path = getattr(snakemake.input, "per_metabolite_metrics", snakemake.input[1])
    best_path = getattr(snakemake.input, "best_models", snakemake.input[2] if len(snakemake.input) > 2 else None)

    if hasattr(snakemake.output, "html"):
        out_path = snakemake.output.html
    elif hasattr(snakemake.output, "report"):
        out_path = snakemake.output.report
    else:
        out_path = snakemake.output[0]

    task = getattr(snakemake.wildcards, "task", Path(out_path).parent.name)

    return global_path, per_met_path, best_path, Path(out_path), task


# -----------------------------
# Load data
# -----------------------------

global_path, per_met_path, best_path, out_path, task = _get_snakemake_io()
out_path.parent.mkdir(parents=True, exist_ok=True)

global_df = pd.read_csv(global_path, sep="\t")
per_met_df = pd.read_parquet(per_met_path)

# best_models.tsv is kept as an input for Snakemake compatibility, but the
# report computes viewer-facing summaries directly from merged_global_metrics.tsv.
try:
    best_df = pd.read_csv(best_path, sep="\t") if best_path is not None else pd.DataFrame()
except Exception:
    best_df = pd.DataFrame()

global_df, per_met_df = _add_compatibility_aliases(global_df, per_met_df)

top_runs = _make_top_runs(global_df, top_n=TOP_N)
best_per_model_r2 = _make_best_per_model_by_r2(global_df)
ranking_comparison = _make_ranking_comparison(global_df, top_runs)
selected_runs = _selected_unique_runs(top_runs)


# -----------------------------
# Plots
# -----------------------------

plot_sections = []

if PLOTLY_AVAILABLE:
    hover_cols = [
        c for c in [
            "method_name",
            "hash",
            "split",
            "test_r2",
            "avg_pearson_per_metabolite",
            "avg_rel_rmse",
            "global_test_pearson",
            "global_test_spearman",
            "test_rmse",
            "test_mae",
            "method_params",
        ]
        if c in global_df.columns
    ]

    if _safe_col(global_df, "avg_rel_rmse") and _safe_col(global_df, "avg_pearson_per_metabolite"):
        fig = px.scatter(
            global_df,
            x="avg_rel_rmse",
            y="avg_pearson_per_metabolite",
            color="method_name" if "method_name" in global_df.columns else None,
            hover_data=hover_cols,
            title="Trade-off: mean per-metabolite Pearson vs mean relative RMSE",
        )
        fig.update_layout(
            xaxis_title="Mean per-metabolite relative RMSE lower is better",
            yaxis_title="Mean per-metabolite Pearson higher is better",
        )
        plot_sections.append(("Trade-off: Pearson vs relative RMSE", _fig_html(fig)))

    if _safe_col(global_df, "avg_rel_rmse") and _safe_col(global_df, "test_r2"):
        fig = px.scatter(
            global_df,
            x="avg_rel_rmse",
            y="test_r2",
            color="method_name" if "method_name" in global_df.columns else None,
            hover_data=hover_cols,
            title="Trade-off: mean per-metabolite R² vs mean relative RMSE",
        )
        fig.update_layout(
            xaxis_title="Mean per-metabolite relative RMSE lower is better",
            yaxis_title="Mean per-metabolite R² higher is better",
        )
        plot_sections.append(("Trade-off: R² vs relative RMSE", _fig_html(fig)))

    if _safe_col(per_met_df, "test_pearson") and "method_name" in per_met_df.columns:
        plot_df = per_met_df[["method_name", "hash", "test_pearson"]].dropna().copy()

        if len(plot_df) > 20000:
            plot_df = plot_df.groupby("method_name", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 2000), random_state=1)
            )

        fig = px.violin(
            plot_df,
            x="method_name",
            y="test_pearson",
            color="method_name",
            box=True,
            points=False,
            title="Per-metabolite Pearson distribution by model",
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Per-metabolite Pearson",
            showlegend=False,
        )
        plot_sections.append(("Per-metabolite Pearson by model", _fig_html(fig)))

    if _safe_col(per_met_df, "test_rel_rmse") and "method_name" in per_met_df.columns:
        plot_df = per_met_df[["method_name", "hash", "test_rel_rmse"]].dropna().copy()

        if len(plot_df) > 20000:
            plot_df = plot_df.groupby("method_name", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 2000), random_state=1)
            )

        fig = px.violin(
            plot_df,
            x="method_name",
            y="test_rel_rmse",
            color="method_name",
            box=True,
            points=False,
            title="Per-metabolite relative RMSE distribution by model",
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Per-metabolite relative RMSE",
            showlegend=False,
        )
        plot_sections.append(("Per-metabolite relative RMSE by model", _fig_html(fig)))
else:
    plot_sections.append(
        ("Plots unavailable", "<p>Plotly is not installed in this environment. Tables were still generated.</p>")
    )


# -----------------------------
# Metabolite-level selected sections
# -----------------------------

metabolite_sections = []

for _, run in selected_runs.iterrows():
    run_hash = str(run.get("hash", ""))
    run_method = str(run.get("method_name", ""))
    run_split = str(run.get("split", ""))
    selected_by = str(run.get("selected_by_label", ""))

    mask = per_met_df["hash"].astype(str).eq(run_hash)

    if "method_name" in per_met_df.columns:
        mask &= per_met_df["method_name"].astype(str).eq(run_method)

    if "split" in per_met_df.columns:
        mask &= per_met_df["split"].astype(str).eq(run_split)

    run_per_met = per_met_df[mask].copy()
    if run_per_met.empty:
        continue

    section_plots = []

    if PLOTLY_AVAILABLE and _safe_col(run_per_met, "test_pearson"):
        fig = px.histogram(
            run_per_met,
            x="test_pearson",
            nbins=min(50, max(5, len(run_per_met))),
            title=f"Per-metabolite Pearson distribution: {run_method} / {run_hash}",
        )
        fig.update_layout(
            xaxis_title="Per-metabolite Pearson",
            yaxis_title="Number of metabolites",
        )
        section_plots.append(_fig_html(fig))

    if PLOTLY_AVAILABLE and _safe_col(run_per_met, "test_rel_rmse"):
        fig = px.histogram(
            run_per_met,
            x="test_rel_rmse",
            nbins=min(50, max(5, len(run_per_met))),
            title=f"Per-metabolite relative RMSE distribution: {run_method} / {run_hash}",
        )
        fig.update_layout(
            xaxis_title="Per-metabolite relative RMSE lower is better",
            yaxis_title="Number of metabolites",
        )
        section_plots.append(_fig_html(fig))

    top_by_pearson = (
        run_per_met.sort_values("test_pearson", ascending=False)
        if "test_pearson" in run_per_met.columns
        else pd.DataFrame()
    )

    best_by_rel_rmse = (
        run_per_met.sort_values("test_rel_rmse", ascending=True)
        if "test_rel_rmse" in run_per_met.columns
        else pd.DataFrame()
    )

    worst_by_pearson = (
        run_per_met.sort_values("test_pearson", ascending=True)
        if "test_pearson" in run_per_met.columns
        else pd.DataFrame()
    )

    section_html = f"""
    <section>
      <h2>Metabolite-level detail: {html.escape(run_method)} / {html.escape(run_hash)}</h2>
      <p class="note">Selected by: <strong>{html.escape(selected_by)}</strong></p>
      {''.join(section_plots)}
      <h3>Top metabolites by per-metabolite Pearson</h3>
      {_display_table(top_by_pearson, columns=METABOLITE_TABLE_COLUMNS, max_rows=20)}
      <h3>Best metabolites by relative RMSE</h3>
      {_display_table(best_by_rel_rmse, columns=METABOLITE_TABLE_COLUMNS, max_rows=20)}
      <h3>Worst metabolites by per-metabolite Pearson</h3>
      {_display_table(worst_by_pearson, columns=METABOLITE_TABLE_COLUMNS, max_rows=20)}
    </section>
    """

    metabolite_sections.append(section_html)


# -----------------------------
# HTML
# -----------------------------

context = {
    "Task": task,
    "Number of runs": len(global_df),
    "Number of per-metabolite rows": len(per_met_df),
    "Top N shown per primary metric": TOP_N,
}

if not global_df.empty:
    for col in ["n_train", "n_test", "n_rna_features", "n_msi_targets", "split"]:
        if col in global_df.columns:
            context[col] = global_df[col].iloc[0]

context_html = "".join(
    f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
    for k, v in context.items()
)

plots_html = "\n".join(
    f"<section><h2>{html.escape(title)}</h2>{content}</section>"
    for title, content in plot_sections
)

metabolite_html = (
    "\n".join(metabolite_sections)
    if metabolite_sections
    else "<p>No selected metabolite-level sections could be generated.</p>"
)

html_doc = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sparkly Pipeline Report - {html.escape(str(task))}</title>
  {'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>' if PLOTLY_AVAILABLE else ''}
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; line-height: 1.45; }}
    h1 {{ margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.35rem; }}
    h3 {{ margin-top: 1.6rem; }}
    .subtitle {{ color: #555; margin-top: 0; }}
    .report-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0 1.25rem 0; }}
    .report-table th, .report-table td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    .report-table th {{ background: #f5f5f5; position: sticky; top: 0; }}
    .note {{ background: #f8f8f8; border-left: 4px solid #999; padding: 10px 12px; }}
    section {{ margin-bottom: 2rem; }}
    .small {{ color: #555; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Sparkly Pipeline Report</h1>
  <p class="subtitle">Task: <strong>{html.escape(str(task))}</strong></p>

  <section>
    <h2>Task overview</h2>
    <table class="report-table">{context_html}</table>
  </section>

  <section>
    <h2>Best runs summary</h2>
    <p class="note">
      The main summary uses three primary metrics: mean per-metabolite R²,
      mean per-metabolite Pearson, and mean per-metabolite relative RMSE.
      Global Pearson/Spearman are shown only as secondary diagnostics.
      Hyperparameters are kept as the right-most column to preserve readability.
    </p>
    {_display_table(top_runs, columns=RUN_TABLE_COLUMNS, max_rows=3 * TOP_N)}
  </section>

  <section>
    <h2>Best configuration per model family</h2>
    <p class="note">One best configuration per model family, selected by mean per-metabolite R².</p>
    {_display_table(best_per_model_r2, columns=PER_MODEL_TABLE_COLUMNS, max_rows=30)}
  </section>

  <section>
    <h2>Consensus ranking across primary metrics</h2>
    <p class="note">
      This table contains the union of runs that appear in the top {TOP_N}
      for at least one primary metric.
    </p>
    {_display_table(ranking_comparison, columns=RANKING_TABLE_COLUMNS, max_rows=50)}
  </section>

  {plots_html}

  {metabolite_html}
</body>
</html>
"""

out_path.write_text(html_doc, encoding="utf-8")
print(f"Saved Sparkly report: {out_path}")
