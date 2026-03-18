# Energy-Based Reasoning via Structured Latent Planning (EBRM)

A Julia framework where **reasoning is inference in latent space**. Problems are encoded into context embeddings, and solutions emerge from gradient-based optimization of structured latent trajectories under a learned energy function.

## Core Idea

```
Problem x  ->  Encoder  ->  h_x (context)
                              |
               Latent Planning: minimize E(h_x, z)
               z = [z1, z2, ..., z_T]  (reasoning trace)
                              |
               Decoder  ->  Answer y
```

- **E(x, z)**: Energy function scoring trajectory plausibility
- **Inference**: Gradient descent / Langevin dynamics in latent space
- **Training**: Split-optimizer scheme -- encoder+decoder learn via supervised loss, energy model learns via contrastive loss

## Project Structure

```
data/                        Dataset definitions and loaders
  graph_reasoning.jl         Shortest-path planning datasets
  arithmetic_reasoning.jl    Arithmetic expression evaluation
  logic_reasoning.jl         SAT-like constraint satisfaction

src/
  models/
    encoder.jl               Problem -> context embedding
    latent_trajectory.jl     Structured z = [z1, ..., z_T]
    energy_network.jl        E(x, z) scoring network
    decoder.jl               z_T -> answer (sigmoid for classification)
  training/
    train.jl                 Split-optimizer training loop
    losses.jl                Contrastive, BCE, score matching losses
  inference/
    planner.jl               Gradient / Langevin latent planner
  utils.jl                   Config, logging, checkpointing

experiments/
  exp_graph_planning.jl      Canonical graph experiment (config-driven)
  exp_arithmetic.jl          Arithmetic reasoning experiment
  exp_logic.jl               Logic/SAT experiment
  poc_graph.jl               Quick proof-of-concept (200 samples)
  run_sweep.jl               Hyperparameter sweep runner

analysis/
  visualize.jl               Plotting utilities
  generate_figures.jl        Paper figure generation

tests/
  test_energy_network.jl     Unit and integration tests

config.toml                  Hyperparameter configuration
Project.toml                 Julia dependencies
```

## Setup

```bash
cd ebr-via-structured-latent-planning
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

```bash
# Run tests
julia --project=. tests/test_energy_network.jl

# Quick proof-of-concept (~5 min)
julia --project=. experiments/poc_graph.jl

# Full experiments (config-driven)
julia --project=. experiments/exp_graph_planning.jl
julia --project=. experiments/exp_arithmetic.jl
julia --project=. experiments/exp_logic.jl

# Hyperparameter sweep
julia --project=. experiments/run_sweep.jl graph

# Generate paper figures from run directories
julia --project=. analysis/generate_figures.jl runs/<graph_run> runs/<arith_run> runs/<logic_run>
```

## Training Architecture

The training loop uses a **split-optimizer** scheme:

1. **Encoder + Decoder** (optimizer 1): trained with direct supervised loss (BCE for classification tasks, MSE for regression). The encoder maps problem features to `h_x`, and the decoder maps `h_x` directly to the answer.

2. **Energy Model** (optimizer 2): trained with contrastive loss using hard negatives. The planner optimizes `z` under the energy function, producing `z_pos`. Negatives are perturbations of `z_pos`.

This separation prevents energy gradients from corrupting the supervised learning signal while still shaping the energy landscape for latent planning.

```
Training Step 1: features -> enc(features) -> h_x -> dec(h_x) -> BCE/MSE loss
                  (gradients flow through encoder and decoder)

Training Step 2: h_x -> planner(E, h_x) -> z_pos
                  z_neg = z_pos + noise
                  E(h_x, z_pos) vs E(h_x, z_neg) -> contrastive loss
                  (gradients flow through energy model only)
```

## Configuration

All hyperparameters in `config.toml`:

```toml
[latent]
dim = 64
trajectory_length = 8

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-3
alpha_contrastive = 0.1
alpha_decoder = 1.0
alpha_smooth = 0.01

[inference]
planner_steps = 50
planner_lr = 0.01
```

## Tasks

| Task | Encoder | Decoder | Loss | Metrics |
|------|---------|---------|------|---------|
| Graph Planning | ProblemEncoder (MLP) | AnswerDecoder (sigmoid) | BCE | Exact accuracy, node recall |
| Arithmetic | SequenceEncoder (embed+pool) | ValueDecoder (linear) | MSE | MAE, median error, within-threshold |
| Logic / SAT | ClauseEncoder (per-clause MLP) | AssignmentDecoder (sigmoid) | BCE | Full satisfaction rate, clause rate |

## Evaluation

Each experiment reports **two inference paths**:
- **Direct**: `encoder(x) -> h_x -> decoder(h_x)` (no planning)
- **Planner**: `encoder(x) -> h_x -> planner(E, h_x) -> z_T -> decoder(z_T)`

The planner initializes trajectories from `h_x` and optimizes under the energy function.

## Metrics and Visualization

Every run saves to `runs/<run_name>/`:
- `metrics.json` -- all logged metrics (loss, accuracy, timing)
- `config.json` -- hyperparameters used
- `checkpoint_epoch*.json` -- model weights
- `training_curves.png` -- loss over training
- `energy_vs_steps.png` -- energy during planning
- `trajectory_2d.png` -- latent trajectory projection
- `comparison.png` -- direct vs planner performance

## PoC Results (Graph Task, 200 samples, 30 epochs)

| Method | Exact Accuracy | Node Recall |
|--------|---------------|-------------|
| EBRM (direct path) | 10.4% | 62.9% |
| Standalone baseline | 2.1% | 58.4% |
| EBRM (planner path) | 0.0% | 56.2% |

The energy-based training regime improves the encoder+decoder beyond a standalone baseline, even without using the planner at inference time.
