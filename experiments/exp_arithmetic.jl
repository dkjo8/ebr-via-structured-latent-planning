"""
Experiment: Arithmetic Reasoning via Energy-Based Reasoning

Trains an EBRM system to evaluate arithmetic expressions using the
split-optimizer training pattern: encoder+decoder learn via supervised
MSE loss, energy model learns via contrastive loss on latent trajectories.
"""

using Random
using Statistics
using Flux
using Plots

include(joinpath(@__DIR__, "..", "src", "training", "train.jl"))
include(joinpath(@__DIR__, "..", "data", "arithmetic_reasoning.jl"))
include(joinpath(@__DIR__, "..", "analysis", "visualize.jl"))

using .Train
using .Train.Encoder
using .Train.EnergyNetwork
using .Train.Decoder
using .Train.Planner
using .Train.Utils
using .Train.Losses
using .ArithmeticReasoning
using .Visualize

const ANSWER_SCALE = 1000f0

function make_arith_prepare_fn()
    function prepare(prob::ArithmeticReasoning.ArithmeticProblem)
        t = ArithmeticReasoning.problem_to_tensors(prob)
        features = t.tokens
        target = Float32[t.answer / ANSWER_SCALE]
        (features, target)
    end
end

function compute_arith_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)
    planner_cfg = Planner.PlannerConfig(; steps=planner_steps, lr=planner_lr, use_langevin=false)
    plan_errors = Float64[]
    direct_errors = Float64[]

    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)

        z_init = 0.01f0 .* randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)

        plan_pred = only(system.decoder(z_opt[:, end])) * ANSWER_SCALE
        push!(plan_errors, abs(plan_pred - prob.answer))

        direct_pred = only(system.decoder(h_x)) * ANSWER_SCALE
        push!(direct_errors, abs(direct_pred - prob.answer))
    end

    n = length(test_data)
    within_1(errs) = count(e -> e < 1.0, errs) / n
    within_10(errs) = count(e -> e < 10.0, errs) / n

    (;
        plan_mae        = mean(plan_errors),
        plan_median     = Statistics.median(plan_errors),
        plan_within_1   = within_1(plan_errors),
        plan_within_10  = within_10(plan_errors),
        direct_mae      = mean(direct_errors),
        direct_median   = Statistics.median(direct_errors),
        direct_within_1 = within_1(direct_errors),
        direct_within_10 = within_10(direct_errors),
    )
end

function run_arithmetic_experiment(; config_path::String=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    Utils.set_seed!(cfg["general"]["seed"])
    rng = MersenneTwister(cfg["general"]["seed"])

    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    ac = cfg["arithmetic_task"]

    @info "Generating arithmetic datasets..."
    train_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_train"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=cfg["general"]["seed"],
    )
    val_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_val"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=cfg["general"]["seed"] + 1,
    )
    test_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_test"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=cfg["general"]["seed"] + 2,
    )
    @info "Datasets" train=length(train_data) val=length(val_data) test=length(test_data)

    prepare_fn = make_arith_prepare_fn()

    vocab_size = ArithmeticReasoning.VOCAB_SIZE
    embed_dim = 32
    encoder = Encoder.SequenceEncoder(vocab_size, embed_dim, latent_dim;
        hidden_dim=cfg["encoder"]["hidden_dim"],
    )
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"],
        n_layers=cfg["energy"]["n_layers"],
    )
    decoder = Decoder.ValueDecoder(latent_dim; hidden_dim=cfg["decoder"]["hidden_dim"])

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)
    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.mse)

    @info "Training EBRM system..."
    state = Train.train!(system, train_data, val_data, prepare_fn, train_config;
        seed=cfg["general"]["seed"],
    )

    planner_steps = cfg["inference"]["planner_steps"]
    planner_lr = cfg["inference"]["planner_lr"]

    @info "Evaluating on test set..."
    m = compute_arith_metrics(system, test_data, prepare_fn, latent_dim, T, planner_steps, planner_lr; rng)

    Utils.log_metric!(state.logger, "test/direct_mae", m.direct_mae; step=state.step)
    Utils.log_metric!(state.logger, "test/direct_median", m.direct_median; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_mae", m.plan_mae; step=state.step)
    Utils.log_metric!(state.logger, "test/plan_median", m.plan_median; step=state.step)
    Utils.save_metrics(state.logger)

    run_dir = state.logger.log_dir
    metrics_data = Visualize.load_metrics(run_dir)
    Visualize.plot_training_curves(metrics_data; save_path=joinpath(run_dir, "training_curves.png"))

    Visualize.plot_comparison(
        ["Direct MAE", "Planner MAE"],
        [m.direct_mae, m.plan_mae];
        ylabel="Mean Absolute Error", title="Arithmetic: Direct vs Planner",
        save_path=joinpath(run_dir, "comparison.png"),
    )

    println()
    println("=" ^ 60)
    println("  ARITHMETIC EXPERIMENT RESULTS")
    println("=" ^ 60)
    println("  Direct path:   MAE=$(round(m.direct_mae;digits=2))  median=$(round(m.direct_median;digits=2))  <1=$(round(m.direct_within_1*100;digits=1))%  <10=$(round(m.direct_within_10*100;digits=1))%")
    println("  Planner path:  MAE=$(round(m.plan_mae;digits=2))  median=$(round(m.plan_median;digits=2))  <1=$(round(m.plan_within_1*100;digits=1))%  <10=$(round(m.plan_within_10*100;digits=1))%")
    println("  Run dir:       $run_dir")
    println("=" ^ 60)

    (; system, state, metrics=m, run_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_arithmetic_experiment()
end
