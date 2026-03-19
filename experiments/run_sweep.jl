"""
Hyperparameter sweep runner for EBRM experiments.

Usage:
  julia --project=. experiments/run_sweep.jl graph
  julia --project=. experiments/run_sweep.jl arithmetic
  julia --project=. experiments/run_sweep.jl logic
  julia --project=. experiments/run_sweep.jl all

Runs a small grid of configurations for the specified task and writes
a summary CSV to analysis/sweep_results_{task}.csv.
"""

using Random
using CSV
using DataFrames

include(joinpath(@__DIR__, "exp_graph_planning.jl"))
include(joinpath(@__DIR__, "exp_arithmetic.jl"))
include(joinpath(@__DIR__, "exp_logic.jl"))

# ── Sweep Grids ──────────────────────────────────────────────

const GRAPH_SWEEP = [
    (latent_dim=32, trajectory_length=4,  planner_steps=10, alpha_contrastive=0.1),
    (latent_dim=32, trajectory_length=8,  planner_steps=10, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=4,  planner_steps=20, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=8,  planner_steps=20, alpha_contrastive=0.05),
    (latent_dim=64, trajectory_length=8,  planner_steps=40, alpha_contrastive=0.2),
]

const ARITH_SWEEP = [
    (latent_dim=32, trajectory_length=4,  planner_steps=10, alpha_contrastive=0.05),
    (latent_dim=32, trajectory_length=8,  planner_steps=20, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=4,  planner_steps=20, alpha_contrastive=0.05),
    (latent_dim=64, trajectory_length=8,  planner_steps=20, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=8,  planner_steps=40, alpha_contrastive=0.05),
]

const LOGIC_SWEEP = [
    (latent_dim=32, trajectory_length=4,  planner_steps=10, alpha_contrastive=0.05),
    (latent_dim=32, trajectory_length=8,  planner_steps=20, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=4,  planner_steps=20, alpha_contrastive=0.1),
    (latent_dim=64, trajectory_length=8,  planner_steps=20, alpha_contrastive=0.2),
    (latent_dim=64, trajectory_length=8,  planner_steps=40, alpha_contrastive=0.1),
]

# ── Config Override ──────────────────────────────────────────

const Utils = Main.Train.Utils

function override_config(base_cfg::Dict, overrides::NamedTuple)
    cfg = deepcopy(base_cfg)
    haskey(overrides, :latent_dim)        && (cfg["latent"]["dim"] = overrides.latent_dim)
    haskey(overrides, :trajectory_length) && (cfg["latent"]["trajectory_length"] = overrides.trajectory_length)
    haskey(overrides, :planner_steps)     && (cfg["inference"]["planner_steps"] = overrides.planner_steps)
    haskey(overrides, :planner_lr)        && (cfg["inference"]["planner_lr"] = overrides.planner_lr)
    haskey(overrides, :alpha_contrastive) && (cfg["training"]["alpha_contrastive"] = overrides.alpha_contrastive)
    haskey(overrides, :alpha_smooth)      && (cfg["training"]["alpha_smooth"] = overrides.alpha_smooth)
    haskey(overrides, :alpha_decoder)     && (cfg["training"]["alpha_decoder"] = overrides.alpha_decoder)
    haskey(overrides, :use_langevin)      && (cfg["inference"]["use_langevin"] = overrides.use_langevin)
    haskey(overrides, :learning_rate)     && (cfg["training"]["learning_rate"] = overrides.learning_rate)
    cfg
end

function apply_sweep_sizes!(cfg::Dict, sweep_cfg::Dict)
    cfg["training"]["epochs"] = get(sweep_cfg, "epochs", 30)
    for task_key in ("graph_task", "arithmetic_task", "logic_task")
        if haskey(cfg, task_key)
            cfg[task_key]["n_train"] = get(sweep_cfg, "n_train", 500)
            cfg[task_key]["n_val"]   = get(sweep_cfg, "n_val", 50)
            cfg[task_key]["n_test"]  = get(sweep_cfg, "n_test", 100)
        end
    end
end

# ── Graph Sweep ──────────────────────────────────────────────

function run_graph_sweep(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    base_cfg = Utils.load_config(config_path)
    if haskey(base_cfg, "sweep")
        apply_sweep_sizes!(base_cfg, base_cfg["sweep"])
    else
        base_cfg["training"]["epochs"] = 30
        base_cfg["graph_task"]["n_train"] = 500
        base_cfg["graph_task"]["n_val"] = 50
        base_cfg["graph_task"]["n_test"] = 100
    end

    rows = []
    for (i, ov) in enumerate(GRAPH_SWEEP)
        @info "Graph sweep $i/$(length(GRAPH_SWEEP))" ov...
        cfg = override_config(base_cfg, ov)
        cfg_path = Utils.write_temp_config(cfg, "sweep_graph_$i")
        result = run_graph_experiment(; config_path=cfg_path)
        m = result.metrics
        push!(rows, (;
            ov...,
            direct_accuracy = round(m.direct_accuracy * 100; digits=1),
            direct_recall   = round(m.direct_recall * 100; digits=1),
            plan_accuracy   = round(m.plan_accuracy * 100; digits=1),
            plan_recall     = round(m.plan_recall * 100; digits=1),
            run_dir         = result.run_dir,
        ))
    end

    df = DataFrame(rows)
    out_path = joinpath(@__DIR__, "..", "analysis", "sweep_results_graph.csv")
    mkpath(dirname(out_path))
    CSV.write(out_path, df)
    @info "Graph sweep results saved to $out_path"
    println(df)
    df
end

# ── Arithmetic Sweep ─────────────────────────────────────────

function run_arith_sweep(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    base_cfg = Utils.load_config(config_path)
    if haskey(base_cfg, "sweep")
        apply_sweep_sizes!(base_cfg, base_cfg["sweep"])
    else
        base_cfg["training"]["epochs"] = 30
        base_cfg["arithmetic_task"]["n_train"] = 500
        base_cfg["arithmetic_task"]["n_val"] = 50
        base_cfg["arithmetic_task"]["n_test"] = 100
    end

    rows = []
    for (i, ov) in enumerate(ARITH_SWEEP)
        @info "Arithmetic sweep $i/$(length(ARITH_SWEEP))" ov...
        cfg = override_config(base_cfg, ov)
        cfg_path = Utils.write_temp_config(cfg, "sweep_arith_$i")
        result = run_arithmetic_experiment(; config_path=cfg_path)
        m = result.metrics
        push!(rows, (;
            ov...,
            direct_mae      = round(m.direct_mae; digits=2),
            direct_median   = round(m.direct_median; digits=2),
            direct_within_1 = round(m.direct_within_1 * 100; digits=1),
            direct_within_10 = round(m.direct_within_10 * 100; digits=1),
            plan_mae        = round(m.plan_mae; digits=2),
            plan_median     = round(m.plan_median; digits=2),
            plan_within_1   = round(m.plan_within_1 * 100; digits=1),
            plan_within_10  = round(m.plan_within_10 * 100; digits=1),
            run_dir         = result.run_dir,
        ))
    end

    df = DataFrame(rows)
    out_path = joinpath(@__DIR__, "..", "analysis", "sweep_results_arithmetic.csv")
    mkpath(dirname(out_path))
    CSV.write(out_path, df)
    @info "Arithmetic sweep results saved to $out_path"
    println(df)
    df
end

# ── Logic Sweep ──────────────────────────────────────────────

function run_logic_sweep(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    base_cfg = Utils.load_config(config_path)
    if haskey(base_cfg, "sweep")
        apply_sweep_sizes!(base_cfg, base_cfg["sweep"])
    else
        base_cfg["training"]["epochs"] = 30
        base_cfg["logic_task"]["n_train"] = 500
        base_cfg["logic_task"]["n_val"] = 50
        base_cfg["logic_task"]["n_test"] = 100
    end

    rows = []
    for (i, ov) in enumerate(LOGIC_SWEEP)
        @info "Logic sweep $i/$(length(LOGIC_SWEEP))" ov...
        cfg = override_config(base_cfg, ov)
        cfg_path = Utils.write_temp_config(cfg, "sweep_logic_$i")
        result = run_logic_experiment(; config_path=cfg_path)
        m = result.metrics
        push!(rows, (;
            ov...,
            direct_sat_rate    = round(m.direct_sat_rate * 100; digits=1),
            direct_clause_rate = round(m.direct_clause_rate * 100; digits=1),
            plan_sat_rate      = round(m.plan_sat_rate * 100; digits=1),
            plan_clause_rate   = round(m.plan_clause_rate * 100; digits=1),
            run_dir            = result.run_dir,
        ))
    end

    df = DataFrame(rows)
    out_path = joinpath(@__DIR__, "..", "analysis", "sweep_results_logic.csv")
    mkpath(dirname(out_path))
    CSV.write(out_path, df)
    @info "Logic sweep results saved to $out_path"
    println(df)
    df
end

# ── CLI Dispatch ─────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    task = length(ARGS) >= 1 ? ARGS[1] : "graph"
    if task == "graph"
        run_graph_sweep()
    elseif task == "arithmetic"
        run_arith_sweep()
    elseif task == "logic"
        run_logic_sweep()
    elseif task == "all"
        @info "Running all sweeps..."
        run_graph_sweep()
        run_arith_sweep()
        run_logic_sweep()
    else
        @error "Unknown task '$task'. Use: graph, arithmetic, logic, all"
    end
end
