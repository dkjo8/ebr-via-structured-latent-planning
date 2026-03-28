"""
Ablation study runner for EBRM experiments.

Usage:
  julia --project=. experiments/run_ablations.jl graph
  julia --project=. experiments/run_ablations.jl arithmetic
  julia --project=. experiments/run_ablations.jl logic

Runs three ablation sets for the specified task:
  A: Energy & planning (contrastive weight, smoothness, no-planner)
  B: Latent structure (trajectory length T)
  C: Optimization dynamics (planner steps, Langevin, step-size)

Outputs CSVs to analysis/ablation_{set}_{task}.csv.
"""

using Random
using CSV
using DataFrames

include(joinpath(@__DIR__, "exp_graph_planning.jl"))
include(joinpath(@__DIR__, "exp_arithmetic.jl"))
include(joinpath(@__DIR__, "exp_logic.jl"))

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
    haskey(overrides, :dual_path_decoder) && (cfg["training"]["dual_path_decoder"] = overrides.dual_path_decoder)
    haskey(overrides, :anchor_weight)     && (cfg["inference"]["anchor_weight"] = overrides.anchor_weight)
    cfg
end

function apply_ablation_sizes!(cfg::Dict)
    abl_cfg = get(cfg, "ablation", Dict())
    cfg["training"]["epochs"] = get(abl_cfg, "epochs", 30)
    for task_key in ("graph_task", "arithmetic_task", "logic_task")
        if haskey(cfg, task_key)
            cfg[task_key]["n_train"] = get(abl_cfg, "n_train", 500)
            cfg[task_key]["n_val"]   = get(abl_cfg, "n_val", 50)
            cfg[task_key]["n_test"]  = get(abl_cfg, "n_test", 100)
        end
    end
end

# ── Ablation Set A: Energy & Planning ────────────────────────

const ABLATION_A = [
    (name="baseline_full",      alpha_contrastive=0.1,  alpha_smooth=0.01, planner_steps=50),
    (name="no_contrastive",     alpha_contrastive=0.0,  alpha_smooth=0.01, planner_steps=50),
    (name="no_smoothness",      alpha_contrastive=0.1,  alpha_smooth=0.0,  planner_steps=50),
    (name="no_planning",        alpha_contrastive=0.1,  alpha_smooth=0.01, planner_steps=0),
    (name="no_energy_at_all",   alpha_contrastive=0.0,  alpha_smooth=0.0,  planner_steps=0),
]

# ── Ablation Set B: Latent Structure ─────────────────────────

const ABLATION_B = [
    (name="T_1",  trajectory_length=1),
    (name="T_2",  trajectory_length=2),
    (name="T_4",  trajectory_length=4),
    (name="T_8",  trajectory_length=8),
    (name="T_12", trajectory_length=12),
]

# ── Ablation Set C: Optimization Dynamics ────────────────────

const ABLATION_C_STEPS = [
    (name="steps_5",   planner_steps=5),
    (name="steps_10",  planner_steps=10),
    (name="steps_25",  planner_steps=25),
    (name="steps_50",  planner_steps=50),
    (name="steps_100", planner_steps=100),
    (name="steps_200", planner_steps=200),
]

const ABLATION_C_LANGEVIN = [
    (name="gd_only",      use_langevin=false),
    (name="langevin_on",  use_langevin=true),
]

const ABLATION_C_LR = [
    (name="plr_0.001", planner_lr=0.001),
    (name="plr_0.005", planner_lr=0.005),
    (name="plr_0.01",  planner_lr=0.01),
    (name="plr_0.05",  planner_lr=0.05),
]

# ── Ablation Set D: Initialization Strategy ──────────────────

const ABLATION_D = [
    (name="init_hx_noise",     planner_steps=50),  # default: z1=h_x, z_{2:T}~N(0,0.01)
    (name="init_all_hx",       planner_steps=50),  # all z_t = h_x + noise (handled by experiment code)
    (name="init_zero",         planner_steps=50),  # z = 0 (handled by experiment code)
]

# ── Ablation Set E: Decoder Training Distribution ────────────

const ABLATION_E = [
    (name="dec_direct_only",   dual_path_decoder=false),
    (name="dec_dual_path",     dual_path_decoder=true),
]

# ── Ablation Set F: Anchor Weight ────────────────────────────

const ABLATION_F = [
    (name="anchor_0.0",   anchor_weight=0.0),
    (name="anchor_0.01",  anchor_weight=0.01),
    (name="anchor_0.1",   anchor_weight=0.1),
    (name="anchor_1.0",   anchor_weight=1.0),
]

# ── Generic Runner ───────────────────────────────────────────

function run_ablation_set(task::String, set_name::String, configs, base_cfg::Dict)
    run_fn = if task == "graph"
        run_graph_experiment
    elseif task == "arithmetic"
        run_arithmetic_experiment
    elseif task == "logic"
        run_logic_experiment
    else
        error("Unknown task: $task")
    end

    rows = []
    for (i, ab) in enumerate(configs)
        ab_name = ab.name
        overrides = NamedTuple(k => v for (k, v) in pairs(ab) if k !== :name)
        @info "Ablation $set_name $i/$(length(configs)): $ab_name" overrides...

        cfg = override_config(base_cfg, overrides)
        cfg_path = Utils.write_temp_config(cfg, "ablation_$(set_name)_$(task)_$i")
        result = run_fn(; config_path=cfg_path)
        m = result.metrics

        row = Dict{Symbol, Any}(:name => ab_name, :run_dir => result.run_dir)
        for (k, v) in pairs(overrides)
            row[k] = v
        end
        for (k, v) in pairs(m)
            row[k] = v isa AbstractFloat ? round(v; digits=4) : v
        end
        push!(rows, (; (k => v for (k, v) in row)...))
    end

    df = DataFrame(rows)
    out_path = joinpath(@__DIR__, "..", "analysis", "ablation_$(set_name)_$(task).csv")
    mkpath(dirname(out_path))
    CSV.write(out_path, df)
    @info "Ablation $set_name ($task) results saved to $out_path"
    println(df)
    df
end

# ── Task-Specific Entry Points ───────────────────────────────

function run_all_ablations(task::String; config_path=joinpath(@__DIR__, "..", "config.toml"))
    base_cfg = Utils.load_config(config_path)
    apply_ablation_sizes!(base_cfg)

    @info "Running ablation set A (energy & planning) for $task"
    df_a = run_ablation_set(task, "A", ABLATION_A, base_cfg)

    @info "Running ablation set B (latent structure) for $task"
    df_b = run_ablation_set(task, "B", ABLATION_B, base_cfg)

    @info "Running ablation set C-steps (planner steps) for $task"
    df_c1 = run_ablation_set(task, "C_steps", ABLATION_C_STEPS, base_cfg)

    @info "Running ablation set C-langevin (GD vs Langevin) for $task"
    df_c2 = run_ablation_set(task, "C_langevin", ABLATION_C_LANGEVIN, base_cfg)

    @info "Running ablation set C-lr (planner step-size) for $task"
    df_c3 = run_ablation_set(task, "C_lr", ABLATION_C_LR, base_cfg)

    @info "Running ablation set D (initialization strategy) for $task"
    df_d = run_ablation_set(task, "D", ABLATION_D, base_cfg)

    @info "Running ablation set E (decoder training distribution) for $task"
    df_e = run_ablation_set(task, "E", ABLATION_E, base_cfg)

    @info "Running ablation set F (anchor weight) for $task"
    df_f = run_ablation_set(task, "F", ABLATION_F, base_cfg)

    @info "All ablations for $task complete."
    (; A=df_a, B=df_b, C_steps=df_c1, C_langevin=df_c2, C_lr=df_c3, D=df_d, E=df_e, F=df_f)
end

# ── CLI ──────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    task = length(ARGS) >= 1 ? ARGS[1] : "graph"
    run_all_ablations(task)
end
