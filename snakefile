import os
import time
import pandas as pd
from scripts.utils import create_tasks_df


def safe_read_csv(path, max_wait=30, interval=1):
    waited = 0
    while waited < max_wait:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, sep="\t")
            except pd.errors.EmptyDataError:
                pass
        time.sleep(interval)
        waited += interval
    raise RuntimeError(f"File '{path}' exists but is still empty or unreadable after {max_wait} seconds.")


os.makedirs("data", exist_ok=True)
tasks_df = create_tasks_df("config.yaml", save="data/tasks.tsv")
tasks_df = safe_read_csv("data/tasks.tsv")

# Keep Snakemake <8 happy by materialising clean lists.
task_list = tasks_df["task"].astype(str).str.strip().tolist()
method_list = tasks_df["method"].astype(str).str.strip().tolist()
hash_list = tasks_df["hash"].astype(str).str.strip().tolist()
unique_tasks = [t.strip() for t in tasks_df["task"].astype(str).unique()]


def task_rows(task):
    return tasks_df[tasks_df["task"].astype(str).str.strip() == str(task).strip()]


def run_partition(wildcards):
    return "gpu_p" if str(wildcards.method).strip() == "xgboost" else "cpu_p"


def run_qos(wildcards):
    return "gpu_normal" if str(wildcards.method).strip() == "xgboost" else "cpu_normal"


def run_gres(wildcards):
    return "--gres=gpu:1" if str(wildcards.method).strip() == "xgboost" else ""


def run_cpus(wildcards):
    return 8 if str(wildcards.method).strip() == "xgboost" else 8


def run_mem_mb(wildcards):
    return 96000 if str(wildcards.method).strip() == "xgboost" else 32000


def run_time(wildcards):
    return "08:00:00" if str(wildcards.method).strip() == "xgboost" else "04:00:00"


rule all:
    input:
        expand(
            "data/reports/{task}/{method}/{hash}/global_metrics.tsv",
            zip,
            task=task_list,
            method=method_list,
            hash=hash_list,
        ),
        expand(
            "data/reports/{task}/{method}/{hash}/per_metabolite_metrics.parquet",
            zip,
            task=task_list,
            method=method_list,
            hash=hash_list,
        ),
        expand(
            "data/reports/{task}/{method}/{hash}/run_metadata.json",
            zip,
            task=task_list,
            method=method_list,
            hash=hash_list,
        ),
        expand("data/reports/{task}/merged_global_metrics.tsv", task=unique_tasks),
        expand("data/reports/{task}/merged_per_metabolite_metrics.parquet", task=unique_tasks),
        expand("data/reports/{task}/best_models.tsv", task=unique_tasks),
        expand("data/reports/{task}/model_report.html", task=unique_tasks),


rule run_method:
    input:
        rna_ds=lambda wildcards: tasks_df.loc[tasks_df["hash"] == wildcards.hash, "input_rna"].values[0],
        metab_ds=lambda wildcards: tasks_df.loc[tasks_df["hash"] == wildcards.hash, "input_metabolomics"].values[0],
    output:
        global_metrics="data/reports/{task}/{method}/{hash}/global_metrics.tsv",
        per_metabolite_metrics="data/reports/{task}/{method}/{hash}/per_metabolite_metrics.parquet",
        run_metadata="data/reports/{task}/{method}/{hash}/run_metadata.json",
    params:
        thisparam=lambda wildcards: tasks_df.loc[tasks_df["hash"] == wildcards.hash, :].iloc[0, :].to_dict(),
    resources:
        partition=run_partition,
        qos=run_qos,
        gres=run_gres,
        cpus=run_cpus,
        mem_mb=run_mem_mb,
        time=run_time,
    script:
        "scripts/run_method.py"


rule merge_global_metrics:
    input:
        lambda wildcards: expand(
            "data/reports/{task}/{method}/{hash}/global_metrics.tsv",
            zip,
            task=[wildcards.task] * len(task_rows(wildcards.task)),
            method=task_rows(wildcards.task)["method"].astype(str).str.strip().tolist(),
            hash=task_rows(wildcards.task)["hash"].astype(str).str.strip().tolist(),
        )
    output:
        tsv="data/reports/{task}/merged_global_metrics.tsv"
    run:
        dfs = [pd.read_csv(file, sep="\t") for file in input if os.path.exists(file)]
        if not dfs:
            raise ValueError(f"No global metrics files found for task {wildcards.task}")
        pd.concat(dfs, ignore_index=True).to_csv(output.tsv, sep="\t", index=False)


rule merge_per_metabolite_metrics:
    input:
        lambda wildcards: expand(
            "data/reports/{task}/{method}/{hash}/per_metabolite_metrics.parquet",
            zip,
            task=[wildcards.task] * len(task_rows(wildcards.task)),
            method=task_rows(wildcards.task)["method"].astype(str).str.strip().tolist(),
            hash=task_rows(wildcards.task)["hash"].astype(str).str.strip().tolist(),
        )
    output:
        parquet="data/reports/{task}/merged_per_metabolite_metrics.parquet"
    run:
        dfs = [pd.read_parquet(file) for file in input if os.path.exists(file)]
        if not dfs:
            raise ValueError(f"No per-metabolite metrics files found for task {wildcards.task}")
        pd.concat(dfs, ignore_index=True).to_parquet(output.parquet, index=False)


rule find_best:
    input:
        tsv="data/reports/{task}/merged_global_metrics.tsv"
    output:
        tsv="data/reports/{task}/best_models.tsv"
    run:
        df = pd.read_csv(input.tsv, sep="\t")

        selection_specs = [
            ("test_pearson", "max"),
            ("avg_rel_rmse", "min"),
            ("test_r2", "max"),
            ("test_rmse", "min"),
        ]

        rows = []
        for metric, direction in selection_specs:
            if metric not in df.columns:
                continue

            valid = df[df[metric].notna()].copy()
            if valid.empty:
                continue

            valid = valid.sort_values(metric, ascending=(direction == "min"))

            # Overall top 10 by metric.
            overall = valid.head(10).copy()
            overall["selection_scope"] = "overall"
            overall["selection_metric"] = metric
            overall["selection_direction"] = direction
            overall["selection_rank"] = range(1, len(overall) + 1)
            rows.append(overall)

            # Best run per model by metric.
            per_model_rows = []
            for _, group in valid.groupby("method_name"):
                per_model_rows.append(group.iloc[0])
            per_model = pd.DataFrame(per_model_rows).copy()
            per_model = per_model.sort_values(metric, ascending=(direction == "min"))
            per_model["selection_scope"] = "per_model"
            per_model["selection_metric"] = metric
            per_model["selection_direction"] = direction
            per_model["selection_rank"] = range(1, len(per_model) + 1)
            rows.append(per_model)

        if not rows:
            raise ValueError("Could not create best_models.tsv because no supported ranking metrics were found.")

        out = pd.concat(rows, ignore_index=True)
        out = out.drop_duplicates(
            subset=["selection_scope", "selection_metric", "method_name", "hash"],
            keep="first",
        )
        out.to_csv(output.tsv, sep="\t", index=False)


rule report:
    input:
        global_metrics="data/reports/{task}/merged_global_metrics.tsv",
        per_metabolite_metrics="data/reports/{task}/merged_per_metabolite_metrics.parquet",
        best_models="data/reports/{task}/best_models.tsv",
    output:
        html="data/reports/{task}/model_report.html"
    script:
        "scripts/model_report.py"
