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

## Outputs

By default, runs write to `runs/<run_name>/`:

- `metrics.json` / `config.json`
- `checkpoint_epoch*.json`
- diagnostic plots (`training_curves.png`, `energy_vs_steps.png`, `trajectory_2d.png`, etc.)

These directories are ignored by git.
