# Running nanochat on LUMI

This folder contains a small LUMI-first launcher for this nanochat fork.

It is meant to get a first run working on LUMI-G. It uses the LUMI AI Factory
ROCm/PyTorch container and avoids the upstream CUDA-oriented `uv sync --extra
gpu` path.

## Quick start

On LUMI:

```bash
cd /scratch/project_462000963/users/$USER
git clone https://github.com/BirgerMoell/nanochat.git
cd nanochat
bash lumi/run-lumi.sh first --project project_462000963
```

Watch the job:

```bash
bash lumi/run-lumi.sh status
bash lumi/run-lumi.sh logs
```

The `first` command submits a short 1-GPU `small-g` Slurm job that:

- creates `.venv-lumi` with access to the container's ROCm PyTorch
- installs nanochat without installing CUDA PyTorch wheels
- downloads a tiny data slice
- trains a small tokenizer
- trains a tiny base model
- runs a short SFT pass
- tries a CLI prompt

Run artifacts go to:

```text
lumi-work/cache/
logs/
.venv-lumi/
```

## Commands

```bash
bash lumi/run-lumi.sh setup --project project_462000963
bash lumi/run-lumi.sh first --project project_462000963
bash lumi/run-lumi.sh pilot --project project_462000963
bash lumi/run-lumi.sh shell --project project_462000963
bash lumi/run-lumi.sh status
bash lumi/run-lumi.sh logs
```

`setup` prepares the Python environment from the login node.

`first` submits a tiny 1-GPU end-to-end job. Start here.

`pilot` submits a short 8-GPU full-node job. Use it only after `first` works.

`shell` opens a short interactive 1-GPU shell.

## Why this exists

The upstream nanochat speedrun is tuned for CUDA on 8x H100. LUMI-G uses AMD
MI250X GPUs through ROCm. The wrapper here:

- uses LUMI's recommended AI Factory container
- reuses the container's ROCm PyTorch instead of installing CUDA wheels
- writes datasets and checkpoints under project scratch
- runs through Slurm instead of the login node
- skips the upstream `--fp8` path, which is not the right first target for
  MI250X

This is a first-run path, not a benchmark configuration.
