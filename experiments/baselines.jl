"""
Standalone neural baselines for EBRM comparison.

Usage:
  julia --project=. experiments/baselines.jl graph
  julia --project=. experiments/baselines.jl arithmetic
  julia --project=. experiments/baselines.jl logic
  julia --project=. experiments/baselines.jl all

Each baseline is a simple encoder->decoder network (no energy model, no planner)
with the same architecture and parameter budget as the EBRM encoder+decoder.
Results are written to analysis/baseline_results.csv.
"""

using Random
using Statistics
using Flux
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "training", "train.jl"))
include(joinpath(@__DIR__, "..", "data", "graph_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "arithmetic_reasoning.jl"))
include(joinpath(@__DIR__, "..", "data", "logic_reasoning.jl"))

using .Train.Encoder
using .Train.Decoder
using .Train.Utils
using .GraphReasoning
using .ArithmeticReasoning
using .LogicReasoning

# ── Generic Baseline Trainer ─────────────────────────────────

function train_baseline!(
    encoder, decoder, train_data, val_data, prepare_fn, loss_fn;
    epochs::Int=100, lr::Float64=1e-3, batch_size::Int=32,
    patience::Int=15, min_delta::Float64=1e-4, seed::Int=42,
)
    rng = MersenneTwister(seed)
    opt = Flux.setup(Adam(lr), (encoder, decoder))
    best_val_loss = Inf
    epochs_no_improve = 0

    for epoch in 1:epochs
        indices = randperm(rng, length(train_data))
        epoch_loss = 0.0
        n = 0

        for idx in indices
            features, target = prepare_fn(train_data[idx])
            l, g = Flux.withgradient(encoder, decoder) do enc, dec
                h = enc(features)
                pred = dec(h)
                loss_fn(pred, target)
            end
            opt, _ = Flux.update!(opt, (encoder, decoder), g)
            epoch_loss += l
            n += 1
        end

        avg_loss = epoch_loss / max(n, 1)

        if epoch % 10 == 0 || epoch == epochs
            val_loss = 0.0
            for i in 1:length(val_data)
                features, target = prepare_fn(val_data[i])
                h = encoder(features)
                pred = decoder(h)
                val_loss += loss_fn(pred, target)
            end
            val_loss /= max(length(val_data), 1)

            @info "Baseline epoch $epoch/$epochs" train_loss=round(avg_loss; digits=4) val_loss=round(val_loss; digits=4)

            if val_loss < best_val_loss - min_delta
                best_val_loss = val_loss
                epochs_no_improve = 0
            else
                epochs_no_improve += 10
            end

            if epochs_no_improve >= patience
                @info "Early stopping at epoch $epoch (patience=$patience)"
                break
            end
        end
    end

    best_val_loss
end

# ── Graph Baseline ───────────────────────────────────────────

function run_graph_baseline(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    seed = cfg["general"]["seed"]
    Utils.set_seed!(seed)
    gc = cfg["graph_task"]
    max_nodes = gc["n_nodes_range"][2]
    latent_dim = cfg["latent"]["dim"]

    train_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_train"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed,
    )
    val_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_val"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed + 1,
    )
    test_data = GraphReasoning.generate_graph_dataset(;
        n=gc["n_test"], n_nodes_range=Tuple(gc["n_nodes_range"]),
        edge_prob=gc["edge_probability"], seed=seed + 2,
    )

    prepare_fn = function(prob)
        t = GraphReasoning.problem_to_tensors(prob; max_n=max_nodes)
        features = vcat(vec(t.node_features), vec(t.adjacency), t.src_onehot, t.dst_onehot)
        target = zeros(Float32, max_nodes)
        for v in prob.shortest_path
            v > 0 && v <= max_nodes && (target[v] = 1f0)
        end
        (features, target)
    end

    input_dim = max_nodes * 3 + max_nodes^2 + 2 * max_nodes
    encoder = Encoder.ProblemEncoder(input_dim, cfg["encoder"]["hidden_dim"], latent_dim;
        n_layers=cfg["encoder"]["n_layers"])
    decoder = Decoder.AnswerDecoder(latent_dim, max_nodes;
        hidden_dim=cfg["decoder"]["hidden_dim"], n_layers=cfg["decoder"]["n_layers"])

    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    train_baseline!(encoder, decoder, train_data, val_data, prepare_fn, Flux.binarycrossentropy;
        epochs, lr, seed)

    correct = 0; total_recall = 0.0
    n = length(test_data)
    for i in 1:n
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        scores = decoder(encoder(features))
        predicted = Set(j for j in 1:length(scores) if scores[j] > 0.5f0)
        ground_truth = Set(prob.shortest_path)
        correct += predicted == ground_truth
        total_recall += length(intersect(predicted, ground_truth)) / max(length(ground_truth), 1)
    end

    metrics = (;
        task = "graph",
        accuracy = round(correct / n * 100; digits=1),
        recall = round(total_recall / n * 100; digits=1),
    )
    @info "Graph baseline" metrics...
    metrics
end

# ── Arithmetic Baseline ──────────────────────────────────────

function run_arith_baseline(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    seed = cfg["general"]["seed"]
    Utils.set_seed!(seed)
    ac = cfg["arithmetic_task"]
    latent_dim = cfg["latent"]["dim"]
    answer_scale = 1000f0

    train_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_train"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed,
    )
    val_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_val"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed + 1,
    )
    test_data = ArithmeticReasoning.generate_arithmetic_dataset(;
        n=ac["n_test"], max_depth=ac["max_depth"],
        max_operand=ac["max_operand"], seed=seed + 2,
    )

    prepare_fn = function(prob)
        t = ArithmeticReasoning.problem_to_tensors(prob)
        (t.tokens, Float32[t.answer / answer_scale])
    end

    vocab_size = ArithmeticReasoning.VOCAB_SIZE
    encoder = Encoder.SequenceEncoder(vocab_size, 32, latent_dim;
        hidden_dim=cfg["encoder"]["hidden_dim"])
    decoder = Decoder.ValueDecoder(latent_dim; hidden_dim=cfg["decoder"]["hidden_dim"])

    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    train_baseline!(encoder, decoder, train_data, val_data, prepare_fn, Flux.mse;
        epochs, lr, seed)

    errors = Float64[]
    n = length(test_data)
    for i in 1:n
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        pred = only(decoder(encoder(features))) * answer_scale
        push!(errors, abs(pred - prob.answer))
    end

    metrics = (;
        task = "arithmetic",
        mae = round(mean(errors); digits=2),
        median_error = round(Statistics.median(errors); digits=2),
        within_1 = round(count(e -> e < 1.0, errors) / n * 100; digits=1),
        within_10 = round(count(e -> e < 10.0, errors) / n * 100; digits=1),
    )
    @info "Arithmetic baseline" metrics...
    metrics
end

# ── Logic Baseline ───────────────────────────────────────────

function run_logic_baseline(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    cfg = Utils.load_config(config_path)
    seed = cfg["general"]["seed"]
    Utils.set_seed!(seed)
    lc = cfg["logic_task"]
    latent_dim = cfg["latent"]["dim"]
    max_vars = lc["n_variables"] + 5
    max_clauses = lc["n_clauses_range"][2] + 6

    train_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_train"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed,
    )
    val_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_val"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed + 1,
    )
    test_data = LogicReasoning.generate_logic_dataset(;
        n=lc["n_test"], n_vars=lc["n_variables"],
        n_clauses_range=Tuple(lc["n_clauses_range"]), seed=seed + 2,
    )

    prepare_fn = function(prob)
        t = LogicReasoning.problem_to_tensors(prob; max_clauses, max_vars)
        (t.clause_matrix, t.target)
    end

    encoder = Encoder.ClauseEncoder(max_vars, latent_dim; hidden_dim=cfg["encoder"]["hidden_dim"])
    decoder = Decoder.AssignmentDecoder(latent_dim, max_vars; hidden_dim=cfg["decoder"]["hidden_dim"])

    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    train_baseline!(encoder, decoder, train_data, val_data, prepare_fn, Flux.binarycrossentropy;
        epochs, lr, seed)

    full_sat = 0; total_clause = 0.0
    n = length(test_data)
    for i in 1:n
        prob = test_data[i]
        features, _ = prepare_fn(prob)
        probs_vec = decoder(encoder(features))
        assignment = BitVector(probs_vec[1:prob.formula.n_vars] .> 0.5f0)
        sat = LogicReasoning.count_satisfied(prob.formula, assignment)
        full_sat += sat == prob.n_clauses
        total_clause += sat / max(prob.n_clauses, 1)
    end

    metrics = (;
        task = "logic",
        sat_rate = round(full_sat / n * 100; digits=1),
        clause_rate = round(total_clause / n * 100; digits=1),
    )
    @info "Logic baseline" metrics...
    metrics
end

# ── Run All & Save ───────────────────────────────────────────

function run_all_baselines(; config_path=joinpath(@__DIR__, "..", "config.toml"))
    results = []
    push!(results, run_graph_baseline(; config_path))
    push!(results, run_arith_baseline(; config_path))
    push!(results, run_logic_baseline(; config_path))

    rows = []
    for r in results
        row = Dict{String, Any}()
        for (k, v) in pairs(r)
            row[string(k)] = v
        end
        push!(rows, row)
    end

    df = DataFrame(rows)
    out_path = joinpath(@__DIR__, "..", "analysis", "baseline_results.csv")
    mkpath(dirname(out_path))
    CSV.write(out_path, df)
    @info "Baseline results saved to $out_path"
    println(df)
    df
end

# ── CLI ──────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    task = length(ARGS) >= 1 ? ARGS[1] : "all"
    if task == "graph"
        run_graph_baseline()
    elseif task == "arithmetic"
        run_arith_baseline()
    elseif task == "logic"
        run_logic_baseline()
    elseif task == "all"
        run_all_baselines()
    else
        @error "Unknown task '$task'. Use: graph, arithmetic, logic, all"
    end
end
