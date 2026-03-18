"""
Experiment: Logical Constraint Reasoning via Energy-Based Reasoning

Trains an EBRM system to solve SAT-like constraint satisfaction problems
using the split-optimizer pattern: encoder+decoder learn via BCE on variable
assignments, energy model learns via contrastive loss on latent trajectories.
"""

using Random
using Statistics
using Flux
using Plots

include(joinpath(@__DIR__, "..", "src", "training", "train.jl"))
include(joinpath(@__DIR__, "..", "data", "logic_reasoning.jl"))
include(joinpath(@__DIR__, "..", "analysis", "visualize.jl"))

using .Train
using .Train.Encoder
using .Train.EnergyNetwork
using .Train.Decoder
using .Train.Planner
using .Train.Utils
using .Train.Losses
using .LogicReasoning
using .Visualize

function make_logic_prepare_fn(max_clauses::Int, max_vars::Int)
    function prepare(prob::LogicReasoning.LogicProblem)
        t = LogicReasoning.problem_to_tensors(prob; max_clauses, max_vars)
        features = t.clause_matrix
        target = t.target
        (features, target)
    end
end

function eval_logic(probs_vec, prob)
    assignment = BitVector(probs_vec[1:prob.formula.n_vars] .> 0.5f0)
    sat = LogicReasoning.count_satisfied(prob.formula, assignment)
    full = sat == prob.n_clauses
    clause_rate = sat / max(prob.n_clauses, 1)
    (; full, clause_rate)
end

function compute_logic_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)
    planner_cfg = Planner.PlannerConfig(; steps=planner_steps, lr=planner_lr, use_langevin=false)
    plan_full = 0; plan_clause = 0.0
    direct_full = 0; direct_clause = 0.0

    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)

        z_init = 0.01f0 .* randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)

        r_plan = eval_logic(system.decoder(z_opt[:, end]), prob)
        plan_full += r_plan.full; plan_clause += r_plan.clause_rate

        r_direct = eval_logic(system.decoder(h_x), prob)
        direct_full += r_direct.full; direct_clause += r_direct.clause_rate
    end
    n = length(test_data)
    (;
        plan_sat_rate    = plan_full / n,
        plan_clause_rate = plan_clause / n,
        direct_sat_rate   = direct_full / n,
        direct_clause_rate = direct_clause / n,
    )
end

function run_logic_experiment(; config_path::String=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    Utils.set_seed!(cfg["general"]["seed"])
    rng = MersenneTwister(cfg["general"]["seed"])

    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    lc = cfg["logic_task"]
    max_vars = lc["n_variables"] + 5
    max_clauses = lc["n_clauses_range"][2] + 6

    @info "Generating logic datasets..."
    train_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_train"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]),
        seed=cfg["general"]["seed"],
    )
    val_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_val"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]),
        seed=cfg["general"]["seed"] + 1,
    )
    test_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_test"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]),
        seed=cfg["general"]["seed"] + 2,
    )
    @info "Datasets" train=length(train_data) val=length(val_data) test=length(test_data)

    prepare_fn = make_logic_prepare_fn(max_clauses, max_vars)

    encoder = Encoder.ClauseEncoder(max_vars, latent_dim; hidden_dim=cfg["encoder"]["hidden_dim"])
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"], n_layers=cfg["energy"]["n_layers"],
    )
    decoder = Decoder.AssignmentDecoder(latent_dim, max_vars; hidden_dim=cfg["decoder"]["hidden_dim"])

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)
    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.binarycrossentropy)

    @info "Training EBRM system..."
    state = Train.train!(system, train_data, val_data, prepare_fn, train_config;
        seed=cfg["general"]["seed"],
    )

    planner_steps = cfg["inference"]["planner_steps"]
    planner_lr = cfg["inference"]["planner_lr"]

    @info "Evaluating on test set..."
    m = compute_logic_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)

    Utils.log_metric!(state.logger, "test/direct_sat_rate", m.direct_sat_rate; step=state.step)
    Utils.log_metric!(state.logger, "test/direct_clause_rate", m.direct_clause_rate; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_sat_rate", m.plan_sat_rate; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_clause_rate", m.plan_clause_rate; step=state.step)
    Utils.save_metrics(state.logger)

    run_dir = state.logger.log_dir
    metrics_data = Visualize.load_metrics(run_dir)
    Visualize.plot_training_curves(metrics_data; save_path=joinpath(run_dir, "training_curves.png"))

    Visualize.plot_comparison(
        ["Direct SAT%", "Planner SAT%"],
        [m.direct_sat_rate * 100, m.plan_sat_rate * 100];
        ylabel="Full Satisfaction Rate (%)", title="Logic: Direct vs Planner",
        save_path=joinpath(run_dir, "comparison.png"),
    )

    println()
    println("=" ^ 60)
    println("  LOGIC EXPERIMENT RESULTS")
    println("=" ^ 60)
    println("  Direct:   sat=$(round(m.direct_sat_rate*100;digits=1))%  clause=$(round(m.direct_clause_rate*100;digits=1))%")
    println("  Planner:  sat=$(round(m.plan_sat_rate*100;digits=1))%  clause=$(round(m.plan_clause_rate*100;digits=1))%")
    println("  Run dir:  $run_dir")
    println("=" ^ 60)

    (; system, state, metrics=m, run_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_logic_experiment()
end
