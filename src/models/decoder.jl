module Decoder

using Flux

export AnswerDecoder, PathDecoder, ValueDecoder, AssignmentDecoder, decode

# ── Generic Answer Decoder ───────────────────────────────────

"""
Maps the final latent state z_T ∈ ℝ^d to an answer y.
The output format depends on the task.
"""
struct AnswerDecoder
    net::Chain
    output_dim::Int
end

Flux.@layer AnswerDecoder

function AnswerDecoder(latent_dim::Int, output_dim::Int; hidden_dim::Int=128, n_layers::Int=2)
    layers = Any[Dense(latent_dim, hidden_dim, relu)]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, relu))
    end
    push!(layers, Dense(hidden_dim, output_dim, sigmoid))
    AnswerDecoder(Chain(layers...), output_dim)
end

decode(dec::AnswerDecoder, z_T::AbstractVector) = dec.net(z_T)
(dec::AnswerDecoder)(z_T) = decode(dec, z_T)

# ── Task-Specific Decoders ───────────────────────────────────

"""
Path decoder for graph planning: outputs a sequence of node logits.
Output shape: (max_nodes, max_path_length).
Each column is a distribution over nodes for that path position.
"""
struct PathDecoder
    step_decoder::Chain
    max_nodes::Int
    max_path_len::Int
end

Flux.@layer PathDecoder

function PathDecoder(latent_dim::Int, max_nodes::Int; max_path_len::Int=20, hidden_dim::Int=128)
    step_decoder = Chain(
        Dense(latent_dim + max_path_len, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, max_nodes),
    )
    PathDecoder(step_decoder, max_nodes, max_path_len)
end

function decode(dec::PathDecoder, z_T::AbstractVector)
    d = length(z_T)
    logits = zeros(Float32, dec.max_nodes, dec.max_path_len)
    for t in 1:dec.max_path_len
        pos = zeros(Float32, dec.max_path_len)
        pos[t] = 1f0
        input = vcat(z_T, pos)
        logits[:, t] .= dec.step_decoder(input)
    end
    logits
end

(dec::PathDecoder)(z_T) = decode(dec, z_T)

"""
Value decoder for arithmetic: outputs a single scalar answer.
"""
struct ValueDecoder
    net::Chain
end

Flux.@layer ValueDecoder

function ValueDecoder(latent_dim::Int; hidden_dim::Int=128)
    net = Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim ÷ 2, relu),
        Dense(hidden_dim ÷ 2, 1),
    )
    ValueDecoder(net)
end

decode(dec::ValueDecoder, z_T::AbstractVector) = dec.net(z_T)
(dec::ValueDecoder)(z_T) = decode(dec, z_T)

"""
Assignment decoder for logic: outputs probabilities for each variable.
"""
struct AssignmentDecoder
    net::Chain
    n_vars::Int
end

Flux.@layer AssignmentDecoder

function AssignmentDecoder(latent_dim::Int, max_vars::Int; hidden_dim::Int=128)
    net = Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, max_vars, sigmoid),
    )
    AssignmentDecoder(net, max_vars)
end

function decode(dec::AssignmentDecoder, z_T::AbstractVector)
    dec.net(z_T)
end

(dec::AssignmentDecoder)(z_T) = decode(dec, z_T)

"""
Convert soft probabilities to a hard BitVector assignment.
"""
function hard_assignment(dec::AssignmentDecoder, z_T::AbstractVector; threshold::Float32=0.5f0)
    probs = decode(dec, z_T)
    BitVector(probs .> threshold)
end

end # module
