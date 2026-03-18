"""
Generate paper-ready figures from completed experiment runs.

Usage:
  julia --project=. analysis/generate_figures.jl <run_dir_graph> <run_dir_arith> <run_dir_logic>

Or call `generate_all_figures(graph_dir, arith_dir, logic_dir)` from Julia.
Outputs to analysis/figures/.
"""

using Plots
using JSON3
using Statistics
using CSV
using DataFrames

include(joinpath(@__DIR__, "visualize.jl"))
using .Visualize

const FIGURES_DIR = joinpath(@__DIR__, "figures")
const ANALYSIS_DIR = @__DIR__

function load_run_metrics(run_dir::String)
    Visualize.load_metrics(run_dir)
end

function load_run_config(run_dir::String)
    path = joinpath(run_dir, "config.json")
    isfile(path) || return Dict()
    JSON3.read(read(path, String))
end

function safe_load_csv(path::String)
    isfile(path) || return nothing
    CSV.read(path, DataFrame)
end

# ── Main Figure Generation ───────────────────────────────────

function generate_all_figures(;
    graph_dir::String="",
    arith_dir::String="",
    logic_dir::String="",
)
    mkpath(FIGURES_DIR)

    # ── Per-Task Training Curves ─────────────────────────────
    for (label, dir, prefix) in [
        ("graph", graph_dir, "fig_graph"),
        ("arithmetic", arith_dir, "fig_arith"),
        ("logic", logic_dir, "fig_logic"),
    ]
        if !isempty(dir) && isdir(dir)
            @info "Generating $label figures from $dir"
            m = load_run_metrics(dir)
            Visualize.plot_training_curves(m;
                save_path=joinpath(FIGURES_DIR, "$(prefix)_training.png"))

            for fname in ("energy_vs_steps.png", "trajectory_2d.png", "comparison.png")
                src = joinpath(dir, fname)
                if isfile(src)
                    dst = joinpath(FIGURES_DIR, "$(prefix)_$(fname)")
                    cp(src, dst; force=true)
                end
            end
        end
    end

    # ── Cross-Task Comparison (all three tasks) ──────────────
    generate_cross_task_figure(graph_dir, arith_dir, logic_dir)

    # ── Ablation Figures ─────────────────────────────────────
    generate_ablation_figures()

    # ── Baseline Comparison ──────────────────────────────────
    generate_baseline_comparison(graph_dir, arith_dir, logic_dir)

    @info "All figures saved to $FIGURES_DIR"
    FIGURES_DIR
end

# ── Cross-Task Comparison ────────────────────────────────────

function generate_cross_task_figure(graph_dir, arith_dir, logic_dir)
    tasks = String[]
    direct_vals = Float64[]
    planner_vals = Float64[]

    if !isempty(graph_dir) && isdir(graph_dir)
        m = load_run_metrics(graph_dir)
        if haskey(m, "test/direct_accuracy") && haskey(m, "test/plan_accuracy")
            push!(tasks, "Graph (Acc%)")
            push!(direct_vals, last(m["test/direct_accuracy"])[2] * 100)
            push!(planner_vals, last(m["test/plan_accuracy"])[2] * 100)
        end
    end

    if !isempty(arith_dir) && isdir(arith_dir)
        m = load_run_metrics(arith_dir)
        if haskey(m, "test/direct_mae") && haskey(m, "test/plan_mae")
            push!(tasks, "Arith (100-MAE)")
            d_mae = last(m["test/direct_mae"])[2]
            p_mae = last(m["test/plan_mae"])[2]
            push!(direct_vals, max(0.0, 100.0 - d_mae))
            push!(planner_vals, max(0.0, 100.0 - p_mae))
        end
    end

    if !isempty(logic_dir) && isdir(logic_dir)
        m = load_run_metrics(logic_dir)
        if haskey(m, "test/direct_sat_rate") && haskey(m, "test/plan_sat_rate")
            push!(tasks, "Logic (SAT%)")
            push!(direct_vals, last(m["test/direct_sat_rate"])[2] * 100)
            push!(planner_vals, last(m["test/plan_sat_rate"])[2] * 100)
        end
    end

    if length(tasks) >= 2
        Visualize.plot_method_comparison(tasks, direct_vals, planner_vals;
            ylabel="Performance",
            title="EBRM: Direct vs Planner Across Tasks",
            save_path=joinpath(FIGURES_DIR, "fig_cross_task.png"),
        )
    end
end

# ── Ablation Figures ─────────────────────────────────────────

function generate_ablation_figures()
    for task in ("graph", "arithmetic", "logic")
        generate_ablation_steps_figure(task)
        generate_ablation_trajectory_length_figure(task)
        generate_ablation_lr_figure(task)
    end
end

function generate_ablation_steps_figure(task::String)
    df = safe_load_csv(joinpath(ANALYSIS_DIR, "ablation_C_steps_$(task).csv"))
    df === nothing && return
    !hasproperty(df, :planner_steps) && return

    steps = Int.(df.planner_steps)
    direct_vals, planner_vals = extract_ablation_metrics(df, task)
    isempty(direct_vals) && return

    ylabel, title_suffix = metric_labels(task)
    Visualize.plot_ablation_line(steps, direct_vals, planner_vals;
        xlabel="Planner Steps", ylabel=ylabel,
        title="Ablation: Planner Steps ($title_suffix)",
        save_path=joinpath(FIGURES_DIR, "fig_ablation_steps_$(task).png"),
    )
end

function generate_ablation_trajectory_length_figure(task::String)
    df = safe_load_csv(joinpath(ANALYSIS_DIR, "ablation_B_$(task).csv"))
    df === nothing && return
    !hasproperty(df, :trajectory_length) && return

    tvals = Int.(df.trajectory_length)
    direct_vals, planner_vals = extract_ablation_metrics(df, task)
    isempty(direct_vals) && return

    ylabel, title_suffix = metric_labels(task)
    Visualize.plot_ablation_line(tvals, direct_vals, planner_vals;
        xlabel="Trajectory Length T", ylabel=ylabel,
        title="Ablation: Trajectory Length ($title_suffix)",
        save_path=joinpath(FIGURES_DIR, "fig_ablation_T_$(task).png"),
    )
end

function generate_ablation_lr_figure(task::String)
    df = safe_load_csv(joinpath(ANALYSIS_DIR, "ablation_C_lr_$(task).csv"))
    df === nothing && return
    !hasproperty(df, :planner_lr) && return

    lrs = Float64.(df.planner_lr)
    direct_vals, planner_vals = extract_ablation_metrics(df, task)
    isempty(direct_vals) && return

    ylabel, title_suffix = metric_labels(task)
    Visualize.plot_ablation_line(lrs, direct_vals, planner_vals;
        xlabel="Planner Learning Rate", ylabel=ylabel,
        title="Ablation: Planner LR ($title_suffix)",
        save_path=joinpath(FIGURES_DIR, "fig_ablation_lr_$(task).png"),
    )
end

function metric_labels(task::String)
    if task == "graph"
        ("Accuracy (%)", "Graph")
    elseif task == "arithmetic"
        ("MAE", "Arithmetic")
    elseif task == "logic"
        ("SAT Rate (%)", "Logic")
    else
        ("Metric", task)
    end
end

function extract_ablation_metrics(df::DataFrame, task::String)
    if task == "graph"
        direct = hasproperty(df, :direct_accuracy) ? Float64.(df.direct_accuracy) .* 100 : Float64[]
        planner = hasproperty(df, :plan_accuracy) ? Float64.(df.plan_accuracy) .* 100 : Float64[]
    elseif task == "arithmetic"
        direct = hasproperty(df, :direct_mae) ? Float64.(df.direct_mae) : Float64[]
        planner = hasproperty(df, :plan_mae) ? Float64.(df.plan_mae) : Float64[]
    elseif task == "logic"
        direct = hasproperty(df, :direct_sat_rate) ? Float64.(df.direct_sat_rate) .* 100 : Float64[]
        planner = hasproperty(df, :plan_sat_rate) ? Float64.(df.plan_sat_rate) .* 100 : Float64[]
    else
        direct = Float64[]
        planner = Float64[]
    end
    (direct, planner)
end

# ── Baseline Comparison ──────────────────────────────────────

function generate_baseline_comparison(graph_dir, arith_dir, logic_dir)
    bl = safe_load_csv(joinpath(ANALYSIS_DIR, "baseline_results.csv"))
    bl === nothing && return

    rows = NamedTuple[]

    graph_bl = filter(r -> r.task == "graph", bl)
    if nrow(graph_bl) > 0 && !isempty(graph_dir) && isdir(graph_dir)
        m = load_run_metrics(graph_dir)
        if haskey(m, "test/direct_accuracy") && haskey(m, "test/plan_accuracy")
            push!(rows, (
                label = "Graph (Acc%)",
                baseline = Float64(graph_bl[1, :accuracy]),
                direct = last(m["test/direct_accuracy"])[2] * 100,
                planner = last(m["test/plan_accuracy"])[2] * 100,
            ))
        end
    end

    logic_bl = filter(r -> r.task == "logic", bl)
    if nrow(logic_bl) > 0 && !isempty(logic_dir) && isdir(logic_dir)
        m = load_run_metrics(logic_dir)
        if haskey(m, "test/direct_sat_rate") && haskey(m, "test/plan_sat_rate")
            push!(rows, (
                label = "Logic (SAT%)",
                baseline = Float64(logic_bl[1, :sat_rate]),
                direct = last(m["test/direct_sat_rate"])[2] * 100,
                planner = last(m["test/plan_sat_rate"])[2] * 100,
            ))
        end
    end

    if !isempty(rows)
        Visualize.plot_results_table(rows;
            ylabel="Performance (%)",
            title="EBRM vs Baseline",
            save_path=joinpath(FIGURES_DIR, "fig_baseline_comparison.png"),
        )
    end
end

# ── CLI ──────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 3
        generate_all_figures(;
            graph_dir=ARGS[1], arith_dir=ARGS[2], logic_dir=ARGS[3])
    elseif length(ARGS) >= 1
        generate_all_figures(; graph_dir=ARGS[1])
    else
        @error "Usage: julia analysis/generate_figures.jl <graph_run_dir> [arith_run_dir] [logic_run_dir]"
    end
end
