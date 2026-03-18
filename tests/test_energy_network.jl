"""
Tests for the EBRM system: energy network, planner, encoder, decoder, and data modules.
"""

using Test
using Random
using Statistics

# ── Load modules ─────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "src", "models", "energy_network.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "encoder.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "decoder.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "latent_trajectory.jl"))
include(joinpath(@__DIR__, "..", "src", "inference", "planner.jl"))
include(joinpath(@__DIR__, "..", "src", "training", "losses.jl"))
include(joinpath(@__DIR__, "..", "data", "graph_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "arithmetic_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "logic_reasoning.jl"))

using .EnergyNetwork
using .Encoder
using .Decoder
using .LatentTrajectory
using .Planner
using .Losses
using .GraphReasoning
using .ArithmeticReasoning
using .LogicReasoning

# ── Energy Network Tests ─────────────────────────────────────

@testset "Energy Network" begin
    d = 16
    T = 4
    model = build_energy_model(d; hidden_dim=32, n_layers=2)

    h_x = randn(Float32, d)
    z = randn(Float32, d, T)

    @testset "forward pass" begin
        e = energy(model, h_x, z)
        @test e isa Float32
        @test isfinite(e)
    end

    @testset "per-step energy" begin
        scores = energy_per_step(model, h_x, z)
        @test length(scores) == T
        @test all(isfinite, scores)
    end

    @testset "differentiable energy" begin
        using Zygote
        e_val, grads = Zygote.withgradient(z) do z_
            model(h_x, z_)
        end
        @test isfinite(e_val)
        @test grads[1] !== nothing
        @test size(grads[1]) == size(z)
    end

    @testset "deterministic" begin
        e1 = energy(model, h_x, z)
        e2 = energy(model, h_x, z)
        @test e1 ≈ e2
    end
end

# ── Latent Trajectory Tests ──────────────────────────────────

@testset "Latent Trajectory" begin
    d = 16
    T = 6
    rng = MersenneTwister(42)

    @testset "initialization" begin
        traj = init_trajectory(rng, d, T)
        @test size(traj) == (d, T)
        @test all(abs.(traj.states) .< 1.0)
    end

    @testset "context initialization" begin
        h_x = randn(rng, Float32, d)
        traj = init_trajectory_from_context(rng, h_x, T)
        @test traj.states[:, 1] ≈ h_x
    end

    @testset "indexing" begin
        traj = init_trajectory(rng, d, T)
        @test length(traj[1]) == d
        traj[1] = zeros(Float32, d)
        @test all(traj[1] .== 0)
    end

    @testset "trajectory to matrix" begin
        traj = init_trajectory(rng, d, T)
        m = trajectory_to_matrix(traj)
        @test m === traj.states
    end
end

# ── Encoder Tests ────────────────────────────────────────────

@testset "Encoders" begin
    d = 16

    @testset "ProblemEncoder" begin
        enc = ProblemEncoder(10, 32, d; n_layers=2)
        x = randn(Float32, 10)
        h = encode(enc, x)
        @test length(h) == d
    end

    @testset "SequenceEncoder" begin
        enc = SequenceEncoder(16, 8, d; hidden_dim=32)
        tokens = [1, 3, 7, 8, 0, 0]
        h = encode(enc, tokens)
        @test length(h) == d
    end

    @testset "ClauseEncoder" begin
        enc = ClauseEncoder(10, d; hidden_dim=32)
        clause_matrix = randn(Float32, 8, 10)
        h = encode(enc, clause_matrix)
        @test length(h) == d
    end
end

# ── Decoder Tests ────────────────────────────────────────────

@testset "Decoders" begin
    d = 16
    z_T = randn(Float32, d)

    @testset "AnswerDecoder" begin
        dec = AnswerDecoder(d, 5; hidden_dim=32)
        y = decode(dec, z_T)
        @test length(y) == 5
        @test all(0 .<= y .<= 1)
    end

    @testset "ValueDecoder" begin
        dec = ValueDecoder(d; hidden_dim=32)
        y = decode(dec, z_T)
        @test length(y) == 1
        @test eltype(y) == Float32
    end

    @testset "AssignmentDecoder" begin
        dec = AssignmentDecoder(d, 5; hidden_dim=32)
        probs = decode(dec, z_T)
        @test length(probs) == 5
        @test all(0 .<= probs .<= 1)
    end

    @testset "PathDecoder" begin
        dec = PathDecoder(d, 10; max_path_len=8, hidden_dim=32)
        logits = decode(dec, z_T)
        @test size(logits) == (10, 8)
    end
end

# ── Planner Tests ────────────────────────────────────────────

@testset "Planner" begin
    d = 16
    T = 4
    model = build_energy_model(d; hidden_dim=32, n_layers=2)
    h_x = randn(Float32, d)
    rng = MersenneTwister(42)

    @testset "basic optimization" begin
        z_init = 0.1f0 .* randn(rng, Float32, d, T)
        config = PlannerConfig(; steps=10, lr=0.01)
        z_opt, trace = optimize_latent(model, h_x, z_init; config)
        @test size(z_opt) == (d, T)
        @test length(trace) > 0
        @test length(trace) <= 10
    end

    @testset "energy decreases" begin
        z_init = randn(rng, Float32, d, T)
        config = PlannerConfig(; steps=20, lr=0.01, use_langevin=false)
        _, trace = optimize_latent(model, h_x, z_init; config)
        if length(trace) >= 2
            @test trace[end] <= trace[1] + 0.1  # allow small tolerance
        end
    end

    @testset "multi-restart" begin
        planner = LatentPlanner(; steps=10, lr=0.01)
        z_opt, trace = multi_restart_plan(planner, model, h_x, d, T; n_restarts=3, rng)
        @test size(z_opt) == (d, T)
    end
end

# ── Loss Tests ───────────────────────────────────────────────

@testset "Losses" begin
    d = 16
    T = 4
    model = build_energy_model(d; hidden_dim=32, n_layers=2)
    h_x = randn(Float32, d)
    z_pos = randn(Float32, d, T)
    z_neg = randn(Float32, d, T)

    @testset "contrastive loss" begin
        l = contrastive_energy_loss(model, h_x, z_pos, z_neg)
        @test l isa Float32
        @test l >= 0
    end

    @testset "smoothness loss" begin
        l = trajectory_smoothness_loss(z_pos)
        @test l isa Float32
        @test l >= 0
    end

    @testset "combined loss" begin
        dec = AnswerDecoder(d, 5; hidden_dim=32)
        target = Float32[1, 0, 1, 0, 0]
        l = combined_loss(model, dec, h_x, z_pos, z_neg, target)
        @test isfinite(l)
    end
end

# ── Data Module Tests ────────────────────────────────────────

@testset "Graph Reasoning Data" begin
    @testset "dataset generation" begin
        ds = generate_graph_dataset(; n=10, n_nodes_range=(5, 8), seed=42)
        @test length(ds) > 0
        @test length(ds) <= 10
    end

    @testset "problem tensors" begin
        ds = generate_graph_dataset(; n=5, n_nodes_range=(5, 8), seed=42)
        prob = ds[1]
        t = GraphReasoning.problem_to_tensors(prob; max_n=10)
        @test size(t.node_features) == (10, 3)
        @test size(t.adjacency) == (10, 10)
        @test sum(t.src_onehot) ≈ 1.0
    end

    @testset "shortest path validity" begin
        ds = generate_graph_dataset(; n=20, seed=42)
        for prob in ds.problems
            @test length(prob.shortest_path) >= 2 || prob.src == prob.dst
            @test prob.shortest_path[1] == prob.src
            @test prob.shortest_path[end] == prob.dst
        end
    end
end

@testset "Arithmetic Reasoning Data" begin
    @testset "dataset generation" begin
        ds = generate_arithmetic_dataset(; n=10, max_depth=3, seed=42)
        @test length(ds) == 10
    end

    @testset "evaluation correctness" begin
        ds = generate_arithmetic_dataset(; n=20, seed=42)
        for prob in ds.problems
            @test prob.answer == evaluate_expression(prob)
        end
    end

    @testset "tensors" begin
        ds = generate_arithmetic_dataset(; n=5, seed=42)
        t = ArithmeticReasoning.problem_to_tensors(ds[1])
        @test length(t.tokens) == 64
        @test t.answer isa Float32
    end
end

@testset "Logic Reasoning Data" begin
    @testset "dataset generation" begin
        ds = generate_logic_dataset(; n=10, n_vars=4, seed=42)
        @test length(ds) == 10
    end

    @testset "planted solutions are valid" begin
        ds = generate_logic_dataset(; n=20, n_vars=5, seed=42)
        for prob in ds.problems
            @test prob.is_satisfiable
            @test check_assignment(prob, prob.solution)
        end
    end

    @testset "tensors" begin
        ds = generate_logic_dataset(; n=5, n_vars=4, seed=42)
        t = LogicReasoning.problem_to_tensors(ds[1]; max_clauses=12, max_vars=8)
        @test size(t.clause_matrix) == (12, 8)
        @test length(t.target) == 8
    end
end

@info "All tests passed."
