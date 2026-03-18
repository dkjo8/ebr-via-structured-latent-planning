"""
Master experiment runner for EBRM.

Usage:
  julia --project=. experiments/run_all.jl

Orchestrates the full experimental pipeline:
  1. Run standalone baselines for all three tasks.
  2. Run hyperparameter sweeps for all three tasks.
  3. Run ablation studies (on graph, the cheapest task).
  4. Run canonical full-scale experiments for all three tasks.
  5. Generate paper-ready figures.
  6. Print summary of best configs and key metrics.
"""

using Random
using DataFrames
using CSV
using Dates

include(joinpath(@__DIR__, "baselines.jl"))
include(joinpath(@__DIR__, "run_sweep.jl"))
include(joinpath(@__DIR__, "run_ablations.jl"))

const CONFIG_PATH = joinpath(@__DIR__, "..", "config.toml")

# ── Step 1: Baselines ────────────────────────────────────────

function step_baselines()
    @info "=" ^ 60
    @info "STEP 1: Running standalone baselines"
    @info "=" ^ 60
    run_all_baselines(; config_path=CONFIG_PATH)
end

# ── Step 2: Sweeps ───────────────────────────────────────────

function step_sweeps()
    @info "=" ^ 60
    @info "STEP 2: Running hyperparameter sweeps"
    @info "=" ^ 60
    df_graph = run_graph_sweep(; config_path=CONFIG_PATH)
    df_arith = run_arith_sweep(; config_path=CONFIG_PATH)
    df_logic = run_logic_sweep(; config_path=CONFIG_PATH)
    (; graph=df_graph, arith=df_arith, logic=df_logic)
end

# ── Step 3: Ablations ────────────────────────────────────────

function step_ablations()
    @info "=" ^ 60
    @info "STEP 3: Running ablation studies (graph task)"
    @info "=" ^ 60
    run_all_ablations("graph"; config_path=CONFIG_PATH)
end

# ── Step 4: Full-Scale Experiments ───────────────────────────

function step_full_experiments()
    @info "=" ^ 60
    @info "STEP 4: Running full-scale canonical experiments"
    @info "=" ^ 60

    @info "Running graph experiment..."
    graph_result = run_graph_experiment(; config_path=CONFIG_PATH)

    @info "Running arithmetic experiment..."
    arith_result = run_arithmetic_experiment(; config_path=CONFIG_PATH)

    @info "Running logic experiment..."
    logic_result = run_logic_experiment(; config_path=CONFIG_PATH)

    (; graph=graph_result, arith=arith_result, logic=logic_result)
end

# ── Step 5: Generate Figures ─────────────────────────────────

function step_figures(results)
    @info "=" ^ 60
    @info "STEP 5: Generating paper-ready figures"
    @info "=" ^ 60

    include(joinpath(@__DIR__, "..", "analysis", "generate_figures.jl"))

    graph_dir = haskey(results, :graph) ? results.graph.run_dir : ""
    arith_dir = haskey(results, :arith) ? results.arith.run_dir : ""
    logic_dir = haskey(results, :logic) ? results.logic.run_dir : ""

    Main.generate_all_figures(;
        graph_dir=graph_dir,
        arith_dir=arith_dir,
        logic_dir=logic_dir,
    )
end

# ── Step 6: Summary ──────────────────────────────────────────

function step_summary(results)
    @info "=" ^ 60
    @info "STEP 6: Experiment Summary"
    @info "=" ^ 60

    println()
    println("=" ^ 70)
    println("  EBRM EXPERIMENTAL RESULTS SUMMARY")
    println("  $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
    println("=" ^ 70)

    if haskey(results, :graph)
        m = results.graph.metrics
        println()
        println("  GRAPH SHORTEST-PATH:")
        println("    Direct:   acc=$(round(m.direct_accuracy*100;digits=1))%  recall=$(round(m.direct_recall*100;digits=1))%")
        println("    Planner:  acc=$(round(m.plan_accuracy*100;digits=1))%  recall=$(round(m.plan_recall*100;digits=1))%")
        println("    Run dir:  $(results.graph.run_dir)")
    end

    if haskey(results, :arith)
        m = results.arith.metrics
        println()
        println("  ARITHMETIC REASONING:")
        println("    Direct:   MAE=$(round(m.direct_mae;digits=2))  <1=$(round(m.direct_within_1*100;digits=1))%  <10=$(round(m.direct_within_10*100;digits=1))%")
        println("    Planner:  MAE=$(round(m.plan_mae;digits=2))  <1=$(round(m.plan_within_1*100;digits=1))%  <10=$(round(m.plan_within_10*100;digits=1))%")
        println("    Run dir:  $(results.arith.run_dir)")
    end

    if haskey(results, :logic)
        m = results.logic.metrics
        println()
        println("  LOGIC CONSTRAINT SATISFACTION:")
        println("    Direct:   sat=$(round(m.direct_sat_rate*100;digits=1))%  clause=$(round(m.direct_clause_rate*100;digits=1))%")
        println("    Planner:  sat=$(round(m.plan_sat_rate*100;digits=1))%  clause=$(round(m.plan_clause_rate*100;digits=1))%")
        println("    Run dir:  $(results.logic.run_dir)")
    end

    bl_path = joinpath(@__DIR__, "..", "analysis", "baseline_results.csv")
    if isfile(bl_path)
        println()
        println("  BASELINES:")
        df = CSV.read(bl_path, DataFrame)
        for row in eachrow(df)
            println("    $(row.task): $(NamedTuple(row))")
        end
    end

    println()
    println("  Output directories:")
    println("    Figures:    analysis/figures/")
    println("    Sweep CSVs: analysis/sweep_results_*.csv")
    println("    Ablations:  analysis/ablation_*.csv")
    println("    Baselines:  analysis/baseline_results.csv")
    println("=" ^ 70)
end

# ── Main ─────────────────────────────────────────────────────

function run_all()
    t0 = time()
    @info "EBRM Full Experimental Pipeline starting at $(Dates.format(now(), "HH:MM:SS"))"

    step_baselines()
    step_sweeps()
    step_ablations()
    results = step_full_experiments()
    step_figures(results)
    step_summary(results)

    elapsed = round((time() - t0) / 60; digits=1)
    @info "Full pipeline completed in $(elapsed) minutes"
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
