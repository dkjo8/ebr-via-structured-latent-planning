# Energy-Based Reasoning via Structured Latent Planning (EBRM)

A Julia codebase where **reasoning is inference in latent space**: problems are encoded into a context vector \(h_x\), and solutions are produced by optimizing a structured latent trajectory \(z_{1:T}\) under a learned energy model \(E(h_x, z)\).

> This repository is research code. Outputs like `runs/` and `analysis/figures/` are generated locally and intentionally not tracked in git.

## Setup

Install [Julia](https://github.com/JuliaLang/juliaup) (Julia \(>= 1.10\)).

Instantiate dependencies:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Quick start

Run the unit/integration tests:

```bash
julia --project=. tests/test_energy_network.jl
```

Run a quick proof-of-concept (graph task):

```bash
julia --project=. experiments/poc_graph.jl
```

Run the canonical experiments (config-driven):

```bash
julia --project=. experiments/exp_graph_planning.jl
julia --project=. experiments/exp_arithmetic.jl
julia --project=. experiments/exp_logic.jl
```

Generate paper figures (reduced settings for speed):

```bash
julia --project=. analysis/generate_paper_figures.jl
```

## What’s in the repo

```
data/                        Synthetic datasets + tensorization
src/                         Models, training loop, inference planner
experiments/                 Canonical runs, sweeps, baselines, ablations
analysis/                    Plotting + paper figure generation
tests/                       Unit/integration tests

config.toml                  Default hyperparameters
Project.toml / Manifest.toml Julia environment
modal_run.py                 Modal runner for GPU figure generation
```

## Core idea

```
Problem x  ->  Encoder  ->  h_x (context)
                              |
               Latent planning: minimize E(h_x, z)
               z = [z1, z2, ..., z_T]  (reasoning trace)
                              |
               Decoder  ->  Answer y
```

- **Energy model**: \(E(h_x, z)\) scores trajectory plausibility.
- **Inference**: gradient descent / Langevin-style updates in latent space.
- **Training**: split-optimizer scheme (encoder+decoder supervised; energy model contrastive).

## Configuration

Most knobs live in `config.toml` (latent dim/length, optimizer settings, planner steps, task dataset sizes).

## Outputs

By default, runs write to `runs/<run_name>/`:

- `metrics.json` / `config.json`
- `checkpoint_epoch*.json`
- diagnostic plots (`training_curves.png`, `energy_vs_steps.png`, `trajectory_2d.png`, etc.)

These directories are ignored by git.

## Citation

If you use this code, please cite the associated work (add bibtex here once the paper/preprint is public).
