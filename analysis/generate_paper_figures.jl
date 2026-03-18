"""
Generate all paper-ready figures for EBRM research.

Trains EBRM systems on all three tasks (graph, arithmetic, logic) with
reduced datasets for speed, then produces every figure type needed:

  1. Energy landscapes around optimal z_T (graph + logic)
  2. Latent trajectories in 2D (graph)
  3. PCA of multiple trajectories across test problems (graph + logic)
  4. Energy vs planning steps (all tasks)
  5. Energy vs clause satisfaction over planning steps (logic)
  6. Error vs final energy scatter (arithmetic)
  7. Training curves (all tasks)
  8. Cross-task direct vs planner comparison
  9. Per-step energy scores along trajectory (graph)

Usage:
  julia --project=. analysis/generate_paper_figures.jl
"""

using Random
using Statistics
using Flux
using Zygote
using Plots

include(joinpath(@__DIR__, "..", "src", "training", "train.jl"))
include(joinpath(@__DIR__, "..", "data", "graph_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "arithmetic_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "logic_reasoning.jl"))
include(joinpath(@__DIR__, "visualize.jl"))

using .Train
using .Train.Encoder
using .Train.EnergyNetwork
using .Train.Decoder
using .Train.Planner
using .Train.Utils
using .Train.Losses
using .Train.GPU
using .GraphReasoning
using .ArithmeticReasoning
using .LogicReasoning
using .Visualize

const FIGS = joinpath(@__DIR__, "figures")
const ANSWER_SCALE = 1000f0

# ── Config helpers ───────────────────────────────────────────

function load_reduced_config(; epochs=30, n_train=300, n_val=30, n_test=50)
    cfg = Utils.load_config(joinpath(@__DIR__, "..", "config.toml"))
    cfg["training"]["epochs"] = epochs
    for tk in ("graph_task", "arithmetic_task", "logic_task")
        haskey(cfg, tk) || continue
        cfg[tk]["n_train"] = n_train
        cfg[tk]["n_val"] = n_val
        cfg[tk]["n_test"] = n_test
    end
    cfg
end

function write_temp_config(cfg::Dict, tag::String)
    path = joinpath(tempdir(), "paperfig_$(tag).toml")
    open(path, "w") do io
        for (section, vals) in cfg
            println(io, "[$section]")
            for (k, v) in vals
                if v isa Vector
                    println(io, "$k = $v")
                elseif v isa String
                    println(io, "$k = \"$v\"")
                elseif v isa Bool
                    println(io, "$k = $(v ? "true" : "false")")
                else
                    println(io, "$k = $v")
                end
            end
            println(io)
        end
    end
    path
end

# ── Prepare functions (same as experiment scripts) ───────────

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

function make_arith_prepare_fn()
    function prepare(prob::ArithmeticReasoning.ArithmeticProblem)
        t = ArithmeticReasoning.problem_to_tensors(prob)
        (t.tokens, Float32[t.answer / ANSWER_SCALE])
    end
end

function make_logic_prepare_fn(max_clauses::Int, max_vars::Int)
    function prepare(prob::LogicReasoning.LogicProblem)
        t = LogicReasoning.problem_to_tensors(prob; max_clauses, max_vars)
        (t.clause_matrix, t.target)
    end
end

cpuarray(x) = GPU.to_cpu(x)
cpuarray(x::Vector) = x

# ── Step-by-step planner with per-step recording ─────────────

function plan_with_traces(energy_model, decoder, h_x, z_init, planner_cfg;
                          record_fn=nothing)
    z = copy(z_init)
    energy_trace = Float64[]
    record_trace = []

    for step in 1:planner_cfg.steps
        e_val, grads = Zygote.withgradient(z) do z_
            energy_model(h_x, z_)
        end
        push!(energy_trace, e_val)

        if record_fn !== nothing
            push!(record_trace, record_fn(z, e_val, step))
        end

        gz = grads[1]
        gz === nothing && break

        grad_norm = sqrt(sum(abs2, gz))
        if grad_norm > planner_cfg.clip_grad
            gz = gz .* (planner_cfg.clip_grad / grad_norm)
        end
        z .= z .- Float32(planner_cfg.lr) .* gz

        if planner_cfg.use_langevin
            noise_scale = Float32(planner_cfg.langevin_noise * sqrt(2.0 * planner_cfg.lr))
            z .+= noise_scale .* randn(Float32, size(z))
        end
    end

    z, energy_trace, record_trace
end

# ══════════════════════════════════════════════════════════════
# GRAPH TASK FIGURES
# ══════════════════════════════════════════════════════════════

function generate_graph_figures(cfg)
    @info "═══ GRAPH: Training and generating figures ═══"
    seed = cfg["general"]["seed"]
    rng = MersenneTwister(seed)
    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    gc = cfg["graph_task"]
    max_nodes = gc["n_nodes_range"][2]

    train_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_train"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed)
    val_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_val"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed + 1)
    test_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_test"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed + 2)

    prepare_fn = make_graph_prepare_fn(max_nodes)
    input_dim = max_nodes * 3 + max_nodes^2 + 2 * max_nodes

    encoder = Encoder.ProblemEncoder(input_dim, cfg["encoder"]["hidden_dim"], latent_dim;
        n_layers=cfg["encoder"]["n_layers"])
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"], n_layers=cfg["energy"]["n_layers"])
    decoder = Decoder.AnswerDecoder(latent_dim, max_nodes;
        hidden_dim=cfg["decoder"]["hidden_dim"], n_layers=cfg["decoder"]["n_layers"])

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)
    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.binarycrossentropy)

    state = Train.train!(system, train_data, val_data, prepare_fn, train_config; seed)

    planner_cfg = Planner.PlannerConfig(;
        steps=cfg["inference"]["planner_steps"],
        lr=cfg["inference"]["planner_lr"], use_langevin=false)

    # ── Fig 1: Energy vs planning steps (multiple test problems) ──
    @info "Graph: energy vs planning steps"
    long_cfg = Planner.PlannerConfig(; steps=planner_cfg.steps * 4,
        lr=planner_cfg.lr, use_langevin=false)
    energy_traces = Vector{Float64}[]
    n_examples = min(5, length(test_data))
    rng_fig = MersenneTwister(seed + 100)

    for i in 1:n_examples
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        _, trace = Planner.optimize_latent(system.energy_model, h_x, z_init; config=long_cfg)
        push!(energy_traces, Float64.(trace))
    end
    Visualize.plot_energy_vs_steps(energy_traces;
        labels=["Problem $i" for i in 1:n_examples],
        title="Graph: Energy During Latent Planning",
        save_path=joinpath(FIGS, "graph_energy_vs_steps.png"))

    # ── Fig 2: Latent trajectory 2D (single problem) ──
    @info "Graph: latent trajectory 2D"
    rng_fig = MersenneTwister(seed + 200)
    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
    z_init[:, 1] .= h_x
    z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=long_cfg)
    Visualize.plot_trajectory_2d(cpuarray(z_opt);
        title="Graph: Latent Trajectory z₁→z_T",
        save_path=joinpath(FIGS, "graph_trajectory_2d.png"))

    # ── Fig 3: Energy landscape around z_T ──
    @info "Graph: energy landscape"
    z_T = cpuarray(z_opt[:, end])
    h_x_cpu = cpuarray(h_x)
    em_cpu = GPU.to_cpu(system.energy_model)
    Visualize.plot_energy_landscape(em_cpu, h_x_cpu, z_T;
        dims=(1, 2), range_size=2.0, n_grid=60,
        title="Graph: Energy Landscape Around z_T",
        save_path=joinpath(FIGS, "graph_energy_landscape_dims12.png"))
    Visualize.plot_energy_landscape(em_cpu, h_x_cpu, z_T;
        dims=(3, 4), range_size=2.0, n_grid=60,
        title="Graph: Energy Landscape (dims 3-4)",
        save_path=joinpath(FIGS, "graph_energy_landscape_dims34.png"))

    # ── Fig 4: PCA of multiple trajectories ──
    @info "Graph: PCA trajectories"
    trajectories = Matrix{Float32}[]
    pca_labels = String[]
    rng_fig = MersenneTwister(seed + 300)
    n_pca = min(8, length(test_data))
    for i in 1:n_pca
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt_i, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
        push!(trajectories, cpuarray(z_opt_i))
        push!(pca_labels, "Problem $i")
    end
    Visualize.plot_planning_trajectory_pca(trajectories;
        labels=pca_labels,
        title="Graph: Latent Trajectories (PCA)",
        save_path=joinpath(FIGS, "graph_pca_trajectories.png"))

    # ── Fig 5: Per-step energy scores along trajectory ──
    @info "Graph: per-step energy scores"
    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    step_scores = EnergyNetwork.energy_per_step(system.energy_model, h_x, z_opt)
    p = plot(1:length(step_scores), Float64.(cpuarray(step_scores));
        xlabel="Trajectory Step t", ylabel="Step Score s(hₓ, zₜ)",
        title="Graph: Per-Step Energy Scores Along Trajectory",
        linewidth=2, marker=:circle, markersize=5, color=:steelblue,
        grid=true, size=(800, 500), legend=false)
    savefig(p, joinpath(FIGS, "graph_per_step_scores.png"))

    # ── Fig 6: Training curves ──
    @info "Graph: training curves"
    metrics = Visualize.load_metrics(state.logger.log_dir)
    Visualize.plot_training_curves(metrics;
        save_path=joinpath(FIGS, "graph_training_curves.png"))

    @info "Graph figures complete"
    (; system, state, test_data, prepare_fn)
end

# ══════════════════════════════════════════════════════════════
# ARITHMETIC TASK FIGURES
# ══════════════════════════════════════════════════════════════

function generate_arith_figures(cfg)
    @info "═══ ARITHMETIC: Training and generating figures ═══"
    seed = cfg["general"]["seed"]
    rng = MersenneTwister(seed)
    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    ac = cfg["arithmetic_task"]

    train_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_train"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed)
    val_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_val"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed + 1)
    test_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_test"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed + 2)

    prepare_fn = make_arith_prepare_fn()

    vocab_size = ArithmeticReasoning.VOCAB_SIZE
    encoder = Encoder.SequenceEncoder(vocab_size, 32, latent_dim;
        hidden_dim=cfg["encoder"]["hidden_dim"])
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"], n_layers=cfg["energy"]["n_layers"])
    decoder = Decoder.ValueDecoder(latent_dim; hidden_dim=cfg["decoder"]["hidden_dim"])

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)
    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.mse)

    state = Train.train!(system, train_data, val_data, prepare_fn, train_config; seed)

    planner_cfg = Planner.PlannerConfig(;
        steps=cfg["inference"]["planner_steps"],
        lr=cfg["inference"]["planner_lr"], use_langevin=false)

    # ── Fig 7: Error vs final energy (scatter) ──
    @info "Arithmetic: error vs energy"
    energies = Float64[]
    errors = Float64[]
    rng_fig = MersenneTwister(seed + 100)

    for i in 1:length(test_data)
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, trace = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)

        final_energy = isempty(trace) ? NaN : last(trace)
        pred = only(system.decoder(z_opt[:, end])) * ANSWER_SCALE
        push!(energies, Float64(final_energy))
        push!(errors, abs(Float64(pred) - Float64(prob.answer)))
    end
    Visualize.plot_error_vs_energy(energies, errors;
        title="Arithmetic: Final Energy vs Prediction Error",
        save_path=joinpath(FIGS, "arith_error_vs_energy.png"))

    # ── Fig 8: Energy vs planning steps (arithmetic) ──
    @info "Arithmetic: energy vs steps"
    long_cfg = Planner.PlannerConfig(; steps=planner_cfg.steps * 4,
        lr=planner_cfg.lr, use_langevin=false)
    arith_traces = Vector{Float64}[]
    n_ex = min(5, length(test_data))
    rng_fig = MersenneTwister(seed + 200)
    for i in 1:n_ex
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        _, trace = Planner.optimize_latent(system.energy_model, h_x, z_init; config=long_cfg)
        push!(arith_traces, Float64.(trace))
    end
    Visualize.plot_energy_vs_steps(arith_traces;
        labels=["Expr $i" for i in 1:n_ex],
        title="Arithmetic: Energy During Latent Planning",
        save_path=joinpath(FIGS, "arith_energy_vs_steps.png"))

    # ── Fig 9: Energy landscape (arithmetic) ──
    @info "Arithmetic: energy landscape"
    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    rng_fig = MersenneTwister(seed + 300)
    z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
    z_init[:, 1] .= h_x
    z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
    z_T = cpuarray(z_opt[:, end])
    em_cpu = GPU.to_cpu(system.energy_model)
    h_x_cpu = cpuarray(h_x)
    Visualize.plot_energy_landscape(em_cpu, h_x_cpu, z_T;
        dims=(1, 2), range_size=2.0, n_grid=60,
        title="Arithmetic: Energy Landscape Around z_T",
        save_path=joinpath(FIGS, "arith_energy_landscape.png"))

    # ── Fig 10: Training curves (arithmetic) ──
    @info "Arithmetic: training curves"
    metrics = Visualize.load_metrics(state.logger.log_dir)
    Visualize.plot_training_curves(metrics;
        save_path=joinpath(FIGS, "arith_training_curves.png"))

    @info "Arithmetic figures complete"
    (; system, state, test_data, prepare_fn, energies, errors)
end

# ══════════════════════════════════════════════════════════════
# LOGIC TASK FIGURES
# ══════════════════════════════════════════════════════════════

function generate_logic_figures(cfg)
    @info "═══ LOGIC: Training and generating figures ═══"
    seed = cfg["general"]["seed"]
    rng = MersenneTwister(seed)
    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    lc = cfg["logic_task"]
    max_vars = lc["n_variables"] + 5
    max_clauses = lc["n_clauses_range"][2] + 6

    train_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_train"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed)
    val_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_val"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed + 1)
    test_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_test"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed + 2)

    prepare_fn = make_logic_prepare_fn(max_clauses, max_vars)

    encoder = Encoder.ClauseEncoder(max_vars, latent_dim; hidden_dim=cfg["encoder"]["hidden_dim"])
    energy_model = EnergyNetwork.build_energy_model(latent_dim;
        hidden_dim=cfg["energy"]["hidden_dim"], n_layers=cfg["energy"]["n_layers"])
    decoder = Decoder.AssignmentDecoder(latent_dim, max_vars; hidden_dim=cfg["decoder"]["hidden_dim"])

    system = Train.EBRMSystem(encoder, energy_model, decoder, latent_dim, T)
    train_config = Train.TrainConfig(cfg; decoder_loss_fn=Flux.binarycrossentropy)

    state = Train.train!(system, train_data, val_data, prepare_fn, train_config; seed)

    planner_cfg = Planner.PlannerConfig(;
        steps=cfg["inference"]["planner_steps"],
        lr=cfg["inference"]["planner_lr"], use_langevin=false)

    # ── Fig 11: Energy vs clause satisfaction over planning steps ──
    @info "Logic: energy vs constraint satisfaction"
    long_cfg = Planner.PlannerConfig(; steps=planner_cfg.steps * 4,
        lr=planner_cfg.lr, use_langevin=false)
    energy_traces = Vector{Float64}[]
    satisfaction_traces = Vector{Float64}[]
    n_ex = min(5, length(test_data))
    rng_fig = MersenneTwister(seed + 100)

    for i in 1:n_ex
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        h_x = system.encoder(features)

        z = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z[:, 1] .= h_x

        e_trace = Float64[]
        s_trace = Float64[]

        for _ in 1:long_cfg.steps
            e_val, grads = Zygote.withgradient(z) do z_
                energy_model(h_x, z_)
            end
            push!(e_trace, e_val)

            z_T = z[:, end]
            probs_vec = system.decoder(z_T)
            assignment = BitVector(probs_vec[1:prob.formula.n_vars] .> 0.5f0)
            sat = LogicReasoning.count_satisfied(prob.formula, assignment)
            push!(s_trace, sat / max(prob.n_clauses, 1))

            gz = grads[1]
            gz === nothing && break

            grad_norm = sqrt(sum(abs2, gz))
            if grad_norm > long_cfg.clip_grad
                gz = gz .* (long_cfg.clip_grad / grad_norm)
            end
            z .= z .- Float32(long_cfg.lr) .* gz
        end

        push!(energy_traces, e_trace)
        push!(satisfaction_traces, s_trace)
    end

    Visualize.plot_energy_vs_constraint_satisfaction(
        energy_traces, satisfaction_traces;
        labels=["Formula $i" for i in 1:n_ex],
        title="Logic: Energy vs Clause Satisfaction During Planning",
        save_path=joinpath(FIGS, "logic_energy_vs_satisfaction.png"))

    # ── Fig 12: Energy vs steps (logic) ──
    @info "Logic: energy vs steps"
    Visualize.plot_energy_vs_steps(energy_traces;
        labels=["Formula $i" for i in 1:n_ex],
        title="Logic: Energy During Latent Planning",
        save_path=joinpath(FIGS, "logic_energy_vs_steps.png"))

    # ── Fig 13: PCA of logic trajectories ──
    @info "Logic: PCA trajectories"
    trajectories = Matrix{Float32}[]
    pca_labels = String[]
    rng_fig = MersenneTwister(seed + 200)
    n_pca = min(8, length(test_data))
    for i in 1:n_pca
        features, _ = prepare_fn(test_data[i])
        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
        push!(trajectories, cpuarray(z_opt))
        push!(pca_labels, "Formula $i")
    end
    Visualize.plot_planning_trajectory_pca(trajectories;
        labels=pca_labels,
        title="Logic: Latent Trajectories (PCA)",
        save_path=joinpath(FIGS, "logic_pca_trajectories.png"))

    # ── Fig 14: Energy landscape (logic) ──
    @info "Logic: energy landscape"
    features, _ = prepare_fn(test_data[1])
    h_x = system.encoder(features)
    rng_fig = MersenneTwister(seed + 300)
    z_init = 0.01f0 .* GPU.device_randn(rng_fig, Float32, latent_dim, T)
    z_init[:, 1] .= h_x
    z_opt, _ = Planner.optimize_latent(system.energy_model, h_x, z_init; config=planner_cfg)
    z_T = cpuarray(z_opt[:, end])
    em_cpu = GPU.to_cpu(system.energy_model)
    h_x_cpu = cpuarray(h_x)
    Visualize.plot_energy_landscape(em_cpu, h_x_cpu, z_T;
        dims=(1, 2), range_size=2.0, n_grid=60,
        title="Logic: Energy Landscape Around z_T",
        save_path=joinpath(FIGS, "logic_energy_landscape.png"))

    # ── Fig 15: Training curves (logic) ──
    @info "Logic: training curves"
    metrics = Visualize.load_metrics(state.logger.log_dir)
    Visualize.plot_training_curves(metrics;
        save_path=joinpath(FIGS, "logic_training_curves.png"))

    @info "Logic figures complete"
    (; system, state, test_data, prepare_fn)
end

# ══════════════════════════════════════════════════════════════
# CROSS-TASK SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════

function generate_cross_task_figure(graph_res, arith_res, logic_res, cfg)
    @info "═══ Generating cross-task summary figure ═══"
    latent_dim = cfg["latent"]["dim"]
    T = cfg["latent"]["trajectory_length"]
    planner_cfg = Planner.PlannerConfig(;
        steps=cfg["inference"]["planner_steps"],
        lr=cfg["inference"]["planner_lr"], use_langevin=false)
    seed = cfg["general"]["seed"]

    # Graph metrics
    rng = MersenneTwister(seed)
    g_plan_correct = 0; g_direct_correct = 0; g_n = length(graph_res.test_data)
    for i in 1:g_n
        prob = graph_res.test_data[i]
        features, _ = graph_res.prepare_fn(prob)
        h_x = graph_res.system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(graph_res.system.energy_model, h_x, z_init; config=planner_cfg)

        plan_scores = graph_res.system.decoder(z_opt[:, end])
        plan_set = Set(j for j in 1:length(plan_scores) if plan_scores[j] > 0.5f0)
        g_plan_correct += plan_set == Set(prob.shortest_path)

        direct_scores = graph_res.system.decoder(h_x)
        direct_set = Set(j for j in 1:length(direct_scores) if direct_scores[j] > 0.5f0)
        g_direct_correct += direct_set == Set(prob.shortest_path)
    end

    # Arithmetic metrics
    rng = MersenneTwister(seed)
    a_plan_err = Float64[]; a_direct_err = Float64[]
    for i in 1:length(arith_res.test_data)
        prob = arith_res.test_data[i]
        features, _ = arith_res.prepare_fn(prob)
        h_x = arith_res.system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(arith_res.system.energy_model, h_x, z_init; config=planner_cfg)
        push!(a_plan_err, abs(only(arith_res.system.decoder(z_opt[:, end])) * ANSWER_SCALE - prob.answer))
        push!(a_direct_err, abs(only(arith_res.system.decoder(h_x)) * ANSWER_SCALE - prob.answer))
    end

    # Logic metrics
    rng = MersenneTwister(seed)
    l_plan_sat = 0; l_direct_sat = 0; l_n = length(logic_res.test_data)
    for i in 1:l_n
        prob = logic_res.test_data[i]
        features, _ = logic_res.prepare_fn(prob)
        h_x = logic_res.system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng, Float32, latent_dim, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(logic_res.system.energy_model, h_x, z_init; config=planner_cfg)

        plan_probs = logic_res.system.decoder(z_opt[:, end])
        plan_assign = BitVector(plan_probs[1:prob.formula.n_vars] .> 0.5f0)
        l_plan_sat += LogicReasoning.count_satisfied(prob.formula, plan_assign) == prob.n_clauses

        direct_probs = logic_res.system.decoder(h_x)
        direct_assign = BitVector(direct_probs[1:prob.formula.n_vars] .> 0.5f0)
        l_direct_sat += LogicReasoning.count_satisfied(prob.formula, direct_assign) == prob.n_clauses
    end

    tasks = ["Graph (Acc%)", "Arith (100-MAE)", "Logic (SAT%)"]
    direct_vals = [
        g_direct_correct / g_n * 100,
        max(0.0, 100.0 - mean(a_direct_err)),
        l_direct_sat / l_n * 100,
    ]
    planner_vals = [
        g_plan_correct / g_n * 100,
        max(0.0, 100.0 - mean(a_plan_err)),
        l_plan_sat / l_n * 100,
    ]

    Visualize.plot_method_comparison(tasks, direct_vals, planner_vals;
        ylabel="Performance",
        title="EBRM: Direct vs Planner Across Tasks",
        save_path=joinpath(FIGS, "cross_task_comparison.png"))

    println()
    println("=" ^ 60)
    println("  CROSS-TASK RESULTS")
    println("=" ^ 60)
    for (t, d, p) in zip(tasks, direct_vals, planner_vals)
        println("  $t:  direct=$(round(d; digits=1))  planner=$(round(p; digits=1))")
    end
    println("=" ^ 60)
end

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

function generate_all_paper_figures()
    mkpath(FIGS)
    cfg = load_reduced_config()

    t0 = time()

    graph_res = generate_graph_figures(cfg)
    arith_res = generate_arith_figures(cfg)
    logic_res = generate_logic_figures(cfg)
    generate_cross_task_figure(graph_res, arith_res, logic_res, cfg)

    elapsed = round((time() - t0) / 60; digits=1)

    println()
    println("=" ^ 60)
    println("  ALL PAPER FIGURES GENERATED")
    println("  Output directory: $FIGS")
    println("  Time: $(elapsed) minutes")
    println("=" ^ 60)
    println()
    println("  Figures produced:")
    for f in sort(readdir(FIGS))
        endswith(f, ".png") && println("    $f")
    end
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_paper_figures()
end
