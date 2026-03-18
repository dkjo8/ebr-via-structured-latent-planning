"""
Experiment: Graph Shortest-Path Planning via Energy-Based Reasoning

Canonical graph experiment driven by config.toml. Uses the split-optimizer
training pattern (enc+dec with BCE, energy model with contrastive loss),
binary membership targets, hₓ-seeded planner, and dual-path evaluation.
"""

using Random
using Statistics
using Flux
using Plots

include(joinpath(@__DIR__, "..", "src", "training", "train.jl"))
include(joinpath(@__DIR__, "..", "data", "graph_reasoning.jl"))
include(joinpath(@__DIR__, "..", "analysis", "visualize.jl"))

using .Train
using .Train.Encoder
using .Train.EnergyNetwork
using .Train.Decoder
using .Train.Planner
using .Train.Utils
using .Train.Losses
using .GraphReasoning
using .Visualize

# ── Shared Helpers ───────────────────────────────────────────

function make_graph_prepare_fn(max_nodes::Int)
    function prepare(prob::GraphReasoning.GraphProblem)
        t = GraphReasoning.problem_to_tensors(prob; max_n=max_nodes)
        features = vcat(vec(t.node_features), vec(t.adjacency), t.src_onehot, t.dst_onehot)
        target = zeros(Float32, max_nodes)
        for v in prob.shortest_path
            v > 0 && v <= max_nodes && (target[v] = 1f0)
        end
        (features, target)
    end
end

function eval_scores(scores, prob)
    predicted_set = Set(j for j in 1:length(scores) if scores[j] > 0.5f0)
    ground_truth = Set(prob.shortest_path)
    exact = predicted_set == ground_truth
    n_path = length(ground_truth)
    n_hit = length(intersect(predicted_set, ground_truth))
    recall = n_hit / max(n_path, 1)
    (; exact, recall)
end

function compute_test_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)
    planner_cfg = Planner.PlannerConfig(; steps=planner_steps, lr=planner_lr, use_langevin=false)
    plan_correct = 0; plan_recall = 0.0
    direct_correct = 0; direct_recall = 0.0

    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)

        z_init = 0.01f0 .* randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)

        r_plan = eval_scores(system.decoder(z_opt[:, end]), prob)
        plan_correct += r_plan.exact; plan_recall += r_plan.recall

        r_direct = eval_scores(system.decoder(h_x), prob)
        direct_correct += r_direct.exact; direct_recall += r_direct.recall
    end
    n = length(test_data)
    (;
        plan_accuracy  = plan_correct / n,  plan_recall  = plan_recall / n,
        direct_accuracy = direct_correct / n, direct_recall = direct_recall / n,
    )
end

function generate_graph_diagnostics(system, test_data, prepare_fn, run_dir, latent_dim, T, planner_steps, planner_lr; seed=42)
    rng = MersenneTwister(seed + 200)
    planner_cfg = Planner.PlannerConfig(; steps=planner_steps * 5, lr=planner_lr, use_langevin=false)

    traces = Vector{Float64}[]
    n_diag = min(5, length(test_data))
    for i in 1:n_diag
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        _, trace = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
        push!(traces, Float64.(trace))
    end
    Visualize.plot_energy_vs_steps(traces;
        labels=["Problem $i" for i in 1:n_diag],
        title="Energy vs Planning Steps",
        save_path=joinpath(run_dir, "energy_vs_steps.png"),
    )

    metrics = Visualize.load_metrics(run_dir)
    Visualize.plot_training_curves(metrics; save_path=joinpath(run_dir, "training_curves.png"))

    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    z_init = 0.01f0 .* randn(rng, Float32, latent_dim, T)
    z_init[:, 1] .= h_x
    z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
    Visualize.plot_trajectory_2d(z_opt;
        title="Latent Trajectory (test problem 1)",
        save_path=joinpath(run_dir, "trajectory_2d.png"),
    )

    @info "Diagnostic plots saved to $run_dir"
end

# ── Main Experiment ──────────────────────────────────────────

function run_graph_experiment(; config_path::String=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    Utils.set_seed!(cfg["general"]["seed"])
    rng = MersenneTwister(cfg["general"]["seed"])

    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    gc = cfg["graph_task"]
    max_nodes = gc["n_nodes_range"][2]

    @info "Generating graph datasets..."
    train_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_train"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=cfg["general"]["seed"],
    )
    val_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_val"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=cfg["general"]["seed"] + 1,
    )
    test_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_test"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=cfg["general"]["seed"] + 2,
    )
    @info "Datasets" train=length(train_data) val=length(val_data) test=length(test_data)

    prepare_fn = make_graph_prepare_fn(max_nodes)

    input_dim = max_nodes * 3 + max_nodes^2 + 2 * max_nodes
    encoder = Encoder.ProblemEncoder(input_dim, cfg["encoder"]["hidden_dim"], latent_dim;
        n_layers=cfg["encoder"]["n_layers"],
    )
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"],
        n_layers=cfg["energy"]["n_layers"],
    )
    decoder = Decoder.AnswerDecoder(latent_dim, max_nodes;
        hidden_dim=cfg["decoder"]["hidden_dim"],
        n_layers=cfg["decoder"]["n_layers"],
    )

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)

    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.binarycrossentropy)

    @info "Training EBRM system..."
    state = Train.train!(system, train_data, val_data, prepare_fn, train_config;
        seed=cfg["general"]["seed"],
    )

    planner_steps = cfg["inference"]["planner_steps"]
    planner_lr = cfg["inference"]["planner_lr"]

    @info "Evaluating on test set..."
    m = compute_test_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)

    Utils.log_metric!(state.logger, "test/direct_accuracy", m.direct_accuracy; step=state.step)
    Utils.log_metric!(state.logger, "test/direct_recall", m.direct_recall; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_accuracy", m.plan_accuracy; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_recall", m.plan_recall; step=state.step)
    Utils.save_metrics(state.logger)

    generate_graph_diagnostics(system, test_data, prepare_fn,
        state.logger.log_dir, latent_dim, T, planner_steps, planner_lr;
        seed=cfg["general"]["seed"],
    )

    run_dir = state.logger.log_dir
    Visualize.plot_comparison(
        ["Direct (enc->dec)", "Planner (z_T->dec)"],
        [m.direct_accuracy * 100, m.plan_accuracy * 100];
        ylabel="Exact Accuracy (%)", title="Graph: Direct vs Planner",
        save_path=joinpath(run_dir, "comparison.png"),
    )

    println()
    println("=" ^ 60)
    println("  GRAPH EXPERIMENT RESULTS")
    println("=" ^ 60)
    println("  Direct path:   acc=$(round(m.direct_accuracy*100;digits=1))%  recall=$(round(m.direct_recall*100;digits=1))%")
    println("  Planner path:  acc=$(round(m.plan_accuracy*100;digits=1))%  recall=$(round(m.plan_recall*100;digits=1))%")
    println("  Run dir:       $run_dir")
    println("=" ^ 60)

    (; system, state, metrics=m, run_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_graph_experiment()
end
