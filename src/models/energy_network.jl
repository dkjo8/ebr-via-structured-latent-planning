module EnergyNetwork

using Flux
using Statistics

export EnergyModel, energy, energy_per_step, build_energy_model

# ── Energy Scoring Network ───────────────────────────────────

"""
Energy function E(x, z) → ℝ that scores how plausible/correct a latent
trajectory z is given problem context h_x.

Architecture:
  1. Per-step scoring: each (h_x, z_t) pair → scalar via shared MLP
  2. Pairwise consistency: adjacent (z_t, z_{t+1}) → scalar via transition scorer
  3. Global: pool per-step scores into a single energy value

Lower energy = better trajectory (inference minimizes E).
"""
struct EnergyModel
    step_scorer::Chain      # scores individual (h_x, z_t) pairs
    transition_scorer::Chain # scores (z_t, z_{t+1}) transitions
    global_scorer::Chain     # aggregates into final energy
    latent_dim::Int
end

Flux.@layer EnergyModel

function build_energy_model(latent_dim::Int; hidden_dim::Int=128, n_layers::Int=3)
    step_layers = Any[Dense(2 * latent_dim, hidden_dim, relu)]
    for _ in 2:n_layers
        push!(step_layers, Dense(hidden_dim, hidden_dim, relu))
    end
    push!(step_layers, Dense(hidden_dim, 1))
    step_scorer = Chain(step_layers...)

    transition_scorer = Chain(
        Dense(2 * latent_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim ÷ 2, relu),
        Dense(hidden_dim ÷ 2, 1),
    )

    global_scorer = Chain(
        Dense(3, hidden_dim ÷ 2, relu),
        Dense(hidden_dim ÷ 2, 1),
    )

    EnergyModel(step_scorer, transition_scorer, global_scorer, latent_dim)
end

"""
Compute scalar energy for context h_x and trajectory z (d × T matrix).

The energy decomposes as:
  E(x, z) = f_global(mean_step_score, mean_transition_score, smoothness)
"""
function energy(model::EnergyModel, h_x::AbstractVector, z::AbstractMatrix)
    d, T = size(z)

    step_scores = Float32[]
    for t in 1:T
        input = vcat(h_x, z[:, t])
        push!(step_scores, only(model.step_scorer(input)))
    end

    trans_scores = Float32[]
    for t in 1:(T-1)
        input = vcat(z[:, t], z[:, t+1])
        push!(trans_scores, only(model.transition_scorer(input)))
    end

    smoothness = T > 1 ? mean(sum((z[:, 2:end] .- z[:, 1:end-1]).^2; dims=1)) : 0f0

    agg_input = Float32[mean(step_scores), T > 1 ? mean(trans_scores) : 0f0, smoothness]
    only(model.global_scorer(agg_input))
end

"""
Return per-step energy scores (useful for visualization).
"""
function energy_per_step(model::EnergyModel, h_x::AbstractVector, z::AbstractMatrix)
    d, T = size(z)
    scores = Vector{Float32}(undef, T)
    for t in 1:T
        input = vcat(h_x, z[:, t])
        scores[t] = only(model.step_scorer(input))
    end
    scores
end

"""
Differentiable energy computation that returns a scalar suitable
for Zygote.gradient with respect to z.
"""
function energy_differentiable(model::EnergyModel, h_x::AbstractVector, z::AbstractMatrix)
    d, T = size(z)

    step_sum = 0f0
    for t in 1:T
        input = vcat(h_x, z[:, t])
        step_sum += only(model.step_scorer(input))
    end

    trans_sum = 0f0
    for t in 1:(T-1)
        input = vcat(z[:, t], z[:, t+1])
        trans_sum += only(model.transition_scorer(input))
    end

    smoothness = T > 1 ? mean(sum((z[:, 2:end] .- z[:, 1:end-1]).^2; dims=1)) : 0f0

    agg_input = Float32[
        step_sum / T,
        T > 1 ? trans_sum / (T - 1) : 0f0,
        smoothness,
    ]
    only(model.global_scorer(agg_input))
end

(model::EnergyModel)(h_x, z) = energy_differentiable(model, h_x, z)

end # module
