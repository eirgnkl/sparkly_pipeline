# Slurm profile for `sparkly_pipeline`

This profile defaults to CPU jobs. The `run_method` rule overrides resources in the Snakefile:

- `xgboost` -> `gpu_p`, `gpu_normal`, `--gres=gpu:1`, 96 GB RAM, 8 h
- all other methods -> `cpu_p`, `cpu_normal`, no GPU request, 32 GB RAM, 4 h

Run from the pipeline root with:

```bash
snakemake --profile profile_slurm -n -p
snakemake --profile profile_slurm
```

The important profile change is that the cluster command uses `{resources.gres}` instead of always using `--gres=gpu:{resources.gpu}`. This allows CPU jobs to omit the GPU request completely.
