module Encoder

using Flux
using Statistics

export ProblemEncoder, GraphEncoder, SequenceEncoder, ClauseEncoder, encode

# ── Base Encoder ─────────────────────────────────────────────

"""
Generic problem encoder: maps variable-size input features
to a fixed-dimensional context embedding h_x ∈ ℝ^d.
"""
struct ProblemEncoder
    net::Chain
    input_dim::Int
    output_dim::Int
end

Flux.@layer ProblemEncoder

function ProblemEncoder(input_dim::Int, hidden_dim::Int, output_dim::Int; n_layers::Int=2)
    layers = []
    push!(layers, Dense(input_dim, hidden_dim, relu))
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, relu))
    end
    push!(layers, Dense(hidden_dim, output_dim))
    ProblemEncoder(Chain(layers...), input_dim, output_dim)
end

function encode(enc::ProblemEncoder, x::AbstractArray)
    enc.net(x)
end

(enc::ProblemEncoder)(x) = encode(enc, x)

# ── Graph Encoder ────────────────────────────────────────────

"""
Encodes graph problems by aggregating node features, adjacency info,
and source/destination indicators into a context vector.
"""
struct GraphEncoder
    node_embed::Dense
    edge_embed::Dense
    query_embed::Dense
    aggregator::Chain
    output_dim::Int
end

Flux.@layer GraphEncoder

function GraphEncoder(max_nodes::Int, latent_dim::Int; hidden_dim::Int=128)
    node_embed = Dense(3, hidden_dim, relu)            # node features (3 dims)
    edge_embed = Dense(max_nodes, hidden_dim, relu)     # row of adjacency
    query_embed = Dense(2 * max_nodes, hidden_dim, relu) # src+dst one-hot
    aggregator = Chain(
        Dense(3 * hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim),
    )
    GraphEncoder(node_embed, edge_embed, query_embed, aggregator, latent_dim)
end

function encode(enc::GraphEncoder, node_features, adjacency, src_oh, dst_oh)
    n_feat = mean_pool(enc.node_embed(node_features'))  # (hidden,)
    e_feat = mean_pool(enc.edge_embed(adjacency))       # (hidden,)
    q_feat = enc.query_embed(vcat(src_oh, dst_oh))      # (hidden,)
    enc.aggregator(vcat(n_feat, e_feat, q_feat))
end

(enc::GraphEncoder)(node_features, adjacency, src_oh, dst_oh) =
    encode(enc, node_features, adjacency, src_oh, dst_oh)

function mean_pool(x::AbstractMatrix)
    vec(mean(x; dims=2))
end

# ── Sequence Encoder ─────────────────────────────────────────

"""
Encodes tokenized arithmetic expressions via embedding + pooling.
"""
struct SequenceEncoder
    embedding::Flux.Embedding
    net::Chain
    output_dim::Int
end

Flux.@layer SequenceEncoder

function SequenceEncoder(vocab_size::Int, embed_dim::Int, latent_dim::Int; hidden_dim::Int=128)
    embedding = Flux.Embedding(vocab_size, embed_dim)
    net = Chain(
        Dense(embed_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim),
    )
    SequenceEncoder(embedding, net, latent_dim)
end

function encode(enc::SequenceEncoder, tokens::AbstractVector{Int})
    mask = tokens .> 0
    any(mask) || return zeros(Float32, enc.output_dim)
    embedded = enc.embedding(tokens[mask])  # (embed_dim, seq_len)
    pooled = vec(mean(embedded; dims=2))
    enc.net(pooled)
end

(enc::SequenceEncoder)(tokens) = encode(enc, tokens)

# ── Clause Encoder ───────────────────────────────────────────

"""
Encodes CNF formulas (clause matrix) into a context vector.
"""
struct ClauseEncoder
    clause_net::Dense
    aggregator::Chain
    output_dim::Int
end

Flux.@layer ClauseEncoder

function ClauseEncoder(max_vars::Int, latent_dim::Int; hidden_dim::Int=128)
    clause_net = Dense(max_vars, hidden_dim, relu)
    aggregator = Chain(
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim),
    )
    ClauseEncoder(clause_net, aggregator, latent_dim)
end

function encode(enc::ClauseEncoder, clause_matrix::AbstractMatrix)
    per_clause = enc.clause_net(clause_matrix')  # (hidden, n_clauses)
    pooled = vec(mean(per_clause; dims=2))
    enc.aggregator(pooled)
end

(enc::ClauseEncoder)(clause_matrix) = encode(enc, clause_matrix)

end # module
