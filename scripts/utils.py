import hashlib
import pandas as pd
import yaml


def create_hash(string: str, digest_size: int = 5):
    return hashlib.blake2b(string.encode("utf-8"), digest_size=digest_size).hexdigest()


def create_tasks_df(config, save=None):
    rows = []

    with open(config, "r") as stream:
        cfg = yaml.safe_load(stream)

    for task, task_dict in cfg["TASKS"].items():
        for method, method_data in task_dict["methods"].items():
            if method_data is None:
                method_params = None
            elif isinstance(method_data, str):
                method_params = method_data
            elif isinstance(method_data, dict):
                method_params = method_data.get("params")
            else:
                raise ValueError(f"Unexpected format for method_data: {method_data}")

            if method_params:
                df_params = pd.read_csv(method_params, sep="\t", index_col=0)
                params_list = [str(row) for row in df_params.to_dict(orient="records")]
            else:
                params_list = ["{}"]

            for param_string in params_list:
                row = {
                    "task": task,
                    "method": method,
                    "params": param_string,
                }

                # Add task-level attributes, e.g. input paths, split, rna_layer, msi_layer.
                for key, value in task_dict.items():
                    if key != "methods":
                        row[key] = value

                hash_basis = "|".join(
                    str(row.get(k, ""))
                    for k in [
                        "task",
                        "method",
                        "params",
                        "input_rna",
                        "input_metabolomics",
                        "split",
                        "rna_layer",
                        "msi_layer",
                    ]
                )
                row["hash"] = create_hash(hash_basis)
                rows.append(row)

    tasks_df = pd.DataFrame(rows)
    tasks_df = tasks_df.drop_duplicates(subset=["task", "method", "params", "hash"])

    if save is not None:
        tasks_df.to_csv(save, sep="\t", index=False)

    return tasks_df
