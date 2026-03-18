"""
Proof-of-Concept: Graph Shortest-Path Planning

Reduced-scale experiment to validate whether the EBRM architecture learns.
Runs fast (~2-5 min), produces diagnostic plots, and compares against a
direct-prediction baseline (encoder → decoder, no latent planning).
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

# ── Configuration (hardcoded for fast iteration) ─────────────

const POC = (
    seed           = 42,
    latent_dim     = 32,
    trajectory_len = 6,
    hidden_dim     = 64,
    n_layers       = 2,
    max_nodes      = 12,
    node_range     = (6, 12),
    edge_prob      = 0.3,
    n_train        = 200,
    n_val          = 30,
    n_test         = 50,
    epochs         = 30,
    batch_size     = 16,
    lr             = 1e-3,
    planner_steps  = 10,
    planner_lr     = 0.01,
    checkpoint_interval = 10,
)

# ── Data ─────────────────────────────────────────────────────

function make_datasets(cfg)
    train = GraphReasoning.generate_graph_dataset(;
        n=cfg.n_train, n_nodes_range=cfg.node_range,
        edge_prob=cfg.edge_prob, seed=cfg.seed,
    )
    val = GraphReasoning.generate_graph_dataset(;
        n=cfg.n_val, n_nodes_range=cfg.node_range,
        edge_prob=cfg.edge_prob, seed=cfg.seed + 1,
    )
    test = GraphReasoning.generate_graph_dataset(;
        n=cfg.n_test, n_nodes_range=cfg.node_range,
        edge_prob=cfg.edge_prob, seed=cfg.seed + 2,
    )
    (; train, val, test)
end

function make_prepare_fn(max_nodes::Int)
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

# ── Model Construction ───────────────────────────────────────

function build_system(cfg)
    input_dim = cfg.max_nodes * 3 + cfg.max_nodes^2 + 2 * cfg.max_nodes
    encoder = Encoder.ProblemEncoder(input_dim, cfg.hidden_dim, cfg.latent_dim; n_layers=cfg.n_layers)
    energy_model = EnergyNetwork.build_energy_model(cfg.latent_dim; hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)
    decoder = Decoder.AnswerDecoder(cfg.latent_dim, cfg.max_nodes; hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)
    Train.EBRMSystem(encoder, energy_model, decoder, cfg.latent_dim, cfg.trajectory_len)
end

function make_train_config(cfg)
    Train.TrainConfig(
        cfg.epochs, cfg.batch_size, cfg.lr, 0.0, cfg.checkpoint_interval,
        0.1f0, 1.0f0, 0.01f0, 1.0f0,
        Planner.PlannerConfig(; steps=cfg.planner_steps, lr=cfg.planner_lr, use_langevin=false),
        Flux.binarycrossentropy,
    )
end

# ── Evaluation ───────────────────────────────────────────────

function eval_scores(scores, prob)
    predicted_set = Set(j for j in 1:length(scores) if scores[j] > 0.5f0)
    ground_truth = Set(prob.shortest_path)
    exact = predicted_set == ground_truth
    n_path = length(ground_truth)
    n_hit = length(intersect(predicted_set, ground_truth))
    recall = n_hit / max(n_path, 1)
    (; exact, recall)
end

function test_metrics(system, test_data, prepare_fn, cfg; rng=MersenneTwister(cfg.seed))
    planner_cfg = Planner.PlannerConfig(; steps=cfg.planner_steps, lr=cfg.planner_lr, use_langevin=false)
    plan_correct = 0; plan_recall = 0.0
    direct_correct = 0; direct_recall = 0.0

    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)

        z_init = 0.01f0 .* randn(rng, Float32, cfg.latent_dim, cfg.trajectory_len)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(
            system.energy_model, h_x, z_init; config=planner_cfg,
        )

        r_plan = eval_scores(system.decoder(z_opt[:, end]), prob)
        plan_correct += r_plan.exact
        plan_recall += r_plan.recall

        r_direct = eval_scores(system.decoder(h_x), prob)
        direct_correct += r_direct.exact
        direct_recall += r_direct.recall
    end
    n = length(test_data)
    (;
        plan_accuracy  = plan_correct / n,
        plan_recall    = plan_recall / n,
        direct_accuracy = direct_correct / n,
        direct_recall   = direct_recall / n,
    )
end

# ── Direct-Prediction Baseline ───────────────────────────────

function train_baseline!(encoder, decoder, train_data, val_data, prepare_fn, cfg)
    rng = MersenneTwister(cfg.seed + 100)
    opt = Flux.setup(Adam(cfg.lr), (encoder, decoder))
    train_losses = Float64[]
    val_losses = Float64[]

    for epoch in 1:cfg.epochs
        indices = randperm(rng, length(train_data))
        epoch_loss = 0.0
        for idx in indices
            features, target = prepare_fn(train_data[idx])
            loss, grads = Flux.withgradient(encoder, decoder) do enc, dec
                h = enc(features)
                pred = dec(h)
                Flux.binarycrossentropy(pred, target)
            end
            opt, _ = Flux.update!(opt, (encoder, decoder), grads)
            epoch_loss += loss
        end
        push!(train_losses, epoch_loss / length(train_data))

        val_loss = 0.0
        for i in 1:length(val_data)
            features, target = prepare_fn(val_data[i])
            val_loss += Flux.binarycrossentropy(decoder(encoder(features)), target)
        end
        push!(val_losses, val_loss / length(val_data))
    end

    (; train_losses, val_losses)
end

function baseline_metrics(encoder, decoder, test_data, prepare_fn)
    exact_correct = 0
    total_recall = 0.0
    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        scores = decoder(encoder(features))
        predicted_set = Set(j for j in 1:length(scores) if scores[j] > 0.5f0)
        ground_truth = Set(prob.shortest_path)
        if predicted_set == ground_truth
            exact_correct += 1
        end
        n_path = length(ground_truth)
        n_hit = length(intersect(predicted_set, ground_truth))
        total_recall += n_hit / max(n_path, 1)
    end
    n = length(test_data)
    (; accuracy=exact_correct / n, node_recall=total_recall / n)
end

# ── Diagnostic Plots ─────────────────────────────────────────

function generate_diagnostics(system, test_data, prepare_fn, state, cfg)
    run_dir = state.logger.log_dir
    rng = MersenneTwister(cfg.seed + 200)
    planner_cfg = Planner.PlannerConfig(; steps=cfg.planner_steps * 5, lr=cfg.planner_lr, use_langevin=false)

    # 1) Energy vs planning steps (5 test problems)
    traces = Vector{Float64}[]
    n_diag = min(5, length(test_data))
    for i in 1:n_diag
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* randn(rng, Float32, cfg.latent_dim, cfg.trajectory_len)
        z_init[:, 1] .= h_x
        _, trace = Planner.optimize_latent(
            system.energy_model, h_x, z_init;
            config=planner_cfg,
        )
        push!(traces, Float64.(trace))
    end
    Visualize.plot_energy_vs_steps(traces;
        labels=["Problem $i" for i in 1:n_diag],
        title="Energy vs Planning Steps (test)",
        save_path=joinpath(run_dir, "energy_vs_steps.png"),
    )

    # 2) Training loss curve
    metrics = Visualize.load_metrics(run_dir)
    Visualize.plot_training_curves(metrics; save_path=joinpath(run_dir, "training_curves.png"))

    # 3) Trajectory 2D projection (first test problem)
    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    z_init = 0.01f0 .* randn(rng, Float32, cfg.latent_dim, cfg.trajectory_len)
    z_init[:, 1] .= h_x
    z_opt, _ = Planner.optimize_latent(
        system.energy_model, h_x, z_init;
        config=planner_cfg,
    )
    Visualize.plot_trajectory_2d(z_opt;
        title="Latent Trajectory (test problem 1)",
        save_path=joinpath(run_dir, "trajectory_2d.png"),
    )

    @info "Diagnostic plots saved to $run_dir"
end

# ── Main ─────────────────────────────────────────────────────

function run_poc()
    cfg = POC
    Utils.set_seed!(cfg.seed)

    @info "Generating datasets..." n_train=cfg.n_train n_val=cfg.n_val n_test=cfg.n_test
    datasets = make_datasets(cfg)
    prepare_fn = make_prepare_fn(cfg.max_nodes)

    # ── EBRM Training ────────────────────────────────────────
    @info "Building EBRM system..." latent_dim=cfg.latent_dim T=cfg.trajectory_len
    system = build_system(cfg)
    train_config = make_train_config(cfg)

    @info "Training EBRM ($(cfg.epochs) epochs, $(cfg.n_train) samples, $(cfg.planner_steps) planner steps)..."
    t0 = time()
    state = Train.train!(system, datasets.train, datasets.val, prepare_fn, train_config; seed=cfg.seed)
    ebrm_time = time() - t0
    @info "EBRM training done" elapsed_seconds=round(ebrm_time; digits=1)

    ebrm_metrics = test_metrics(system, datasets.test, prepare_fn, cfg)
    ebrm_acc = ebrm_metrics.direct_accuracy
    @info "EBRM test (planner)" accuracy=round(ebrm_metrics.plan_accuracy * 100; digits=1) node_recall=round(ebrm_metrics.plan_recall * 100; digits=1)
    @info "EBRM test (direct)" accuracy=round(ebrm_metrics.direct_accuracy * 100; digits=1) node_recall=round(ebrm_metrics.direct_recall * 100; digits=1)

    # ── Baseline Training ────────────────────────────────────
    @info "Training direct-prediction baseline..."
    input_dim = cfg.max_nodes * 3 + cfg.max_nodes^2 + 2 * cfg.max_nodes
    bl_encoder = Encoder.ProblemEncoder(input_dim, cfg.hidden_dim, cfg.latent_dim; n_layers=cfg.n_layers)
    bl_decoder = Decoder.AnswerDecoder(cfg.latent_dim, cfg.max_nodes; hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)

    t0 = time()
    bl_results = train_baseline!(bl_encoder, bl_decoder, datasets.train, datasets.val, prepare_fn, cfg)
    bl_time = time() - t0
    @info "Baseline training done" elapsed_seconds=round(bl_time; digits=1)

    bl_metrics = baseline_metrics(bl_encoder, bl_decoder, datasets.test, prepare_fn)
    bl_acc = bl_metrics.accuracy
    @info "Baseline test" accuracy=round(bl_acc * 100; digits=1) node_recall=round(bl_metrics.node_recall * 100; digits=1)

    # ── Diagnostics ──────────────────────────────────────────
    Utils.save_metrics(state.logger)
    generate_diagnostics(system, datasets.test, prepare_fn, state, cfg)

    # ── Summary ──────────────────────────────────────────────
    run_dir = state.logger.log_dir
    Visualize.plot_comparison(
        ["EBRM (planning)", "Direct (no planning)"],
        [ebrm_acc * 100, bl_acc * 100];
        ylabel="Node-set Accuracy (%)",
        title="EBRM vs Direct Prediction",
        save_path=joinpath(run_dir, "comparison.png"),
    )

    println()
    println("=" ^ 60)
    println("  PROOF-OF-CONCEPT RESULTS")
    println("=" ^ 60)
    println("  EBRM (planner path):")
    println("    Exact accuracy:  $(round(ebrm_metrics.plan_accuracy * 100; digits=1))%")
    println("    Node recall:     $(round(ebrm_metrics.plan_recall * 100; digits=1))%")
    println("  EBRM (direct path, enc->dec):")
    println("    Exact accuracy:  $(round(ebrm_metrics.direct_accuracy * 100; digits=1))%")
    println("    Node recall:     $(round(ebrm_metrics.direct_recall * 100; digits=1))%")
    println("    Time:            $(round(ebrm_time; digits=1))s")
    println("  Baseline (separate enc+dec):")
    println("    Exact accuracy:  $(round(bl_acc * 100; digits=1))%")
    println("    Node recall:     $(round(bl_metrics.node_recall * 100; digits=1))%")
    println("    Time:            $(round(bl_time; digits=1))s")
    println("  Plots saved to:   $run_dir/")
    println("=" ^ 60)

    (; system, state, ebrm_acc, bl_acc, ebrm_metrics, bl_metrics, run_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_poc()
end
