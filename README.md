# Energy-Based Reasoning via Structured Latent Planning (EBRM)

> 🚧 WORK IN PROGRESS 🚧

Julia research code for **reasoning as inference in latent space**: encode a problem into a context vector \(h_x\), optimize a structured latent trajectory \(z_{1:T}\) under a learned energy \(E(h_x, z)\), and decode an answer. Generated outputs (`runs/`, local figures) are not tracked in git.

There is no `LICENSE` file in this repository yet; default copyright applies until terms are added.

---

## Setup

Install [Julia](https://github.com/JuliaLang/juliaup) (Julia \(>= 1.10\)).

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

---

## Quick start

**Tests**

```bash
julia --project=. tests/test_energy_network.jl
```

**Proof-of-concept (graph)**

```bash
julia --project=. experiments/poc_graph.jl
```

**Canonical experiments** (config-driven)

```bash
julia --project=. experiments/exp_graph_planning.jl
julia --project=. experiments/exp_arithmetic.jl
julia --project=. experiments/exp_logic.jl
```

**Paper figures** (reduced settings for speed)

```bash
julia --project=. analysis/generate_paper_figures.jl
```

**Convenience scripts**

```bash
./scripts/run_tests.sh
./scripts/run_all.sh
./scripts/run_paper_figures.sh
```

### Hyperparameter sweeps

Grid search over a small set of configs per task; writes CSV summaries under `analysis/` (e.g. `sweep_results_graph.csv`).

```bash
julia --project=. experiments/run_sweep.jl graph
julia --project=. experiments/run_sweep.jl arithmetic
julia --project=. experiments/run_sweep.jl logic
julia --project=. experiments/run_sweep.jl all
```

### Ablation studies

Task-specific ablation grids (energy/planning, latent structure, optimizer); CSVs under `analysis/` (e.g. `ablation_A_graph.csv`).

```bash
julia --project=. experiments/run_ablations.jl graph
julia --project=. experiments/run_ablations.jl arithmetic
julia --project=. experiments/run_ablations.jl logic
```

### Standalone baselines

Encoder→decoder baselines (no energy model, no planner) for comparison; results in `analysis/baseline_results.csv`. Also run from `experiments/run_all.jl`.

```bash
julia --project=. experiments/baselines.jl graph
julia --project=. experiments/baselines.jl arithmetic
julia --project=. experiments/baselines.jl logic
julia --project=. experiments/baselines.jl all
```

---

## Repository layout

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

---

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

---

## Configuration

Most knobs live in `config.toml` (latent dim/length, optimizer settings, planner steps, task dataset sizes).

---

## Modal (cloud GPU)

Optional: run training/figure generation on a Modal GPU using `modal_run.py` (see the module docstring for entrypoints). After syncing the volume, artifacts typically appear under `modal_output/` locally.

```bash
modal run modal_run.py
```

---

## Outputs

By default, runs write to `runs/<run_name>/`:

- `metrics.json` / `config.json`
- `checkpoint_epoch*.json`
- diagnostic plots (`training_curves.png`, `energy_vs_steps.png`, `trajectory_2d.png`, etc.)

These directories are ignored by git.

Modal GPU runs write under `modal_output/` on the host after download; that path is also ignored.
