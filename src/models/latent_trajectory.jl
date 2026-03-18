module LatentTrajectory

using Flux
using Random
using LinearAlgebra

export StructuredTrajectory, init_trajectory, init_trajectory_from_context,
       trajectory_to_matrix, TrajectoryTransition, step_trajectory

# ── Trajectory Representation ────────────────────────────────

"""
A structured latent trajectory z = [z₁, z₂, …, z_T] where z_t ∈ ℝ^d.

Stored as a (d × T) matrix for efficient batched operations.
The trajectory represents the latent "reasoning trace" that the
energy function scores and the planner optimizes.
"""
struct StructuredTrajectory
    states::Matrix{Float32}   # (d, T)
    dim::Int
    length::Int
end

Base.size(traj::StructuredTrajectory) = (traj.dim, traj.length)
Base.getindex(traj::StructuredTrajectory, t::Int) = traj.states[:, t]

function Base.setindex!(traj::StructuredTrajectory, v::AbstractVector, t::Int)
    traj.states[:, t] .= v
end

"""
Initialize a trajectory with small random noise.
"""
function init_trajectory(rng::AbstractRNG, d::Int, T::Int; scale::Float32=0.01f0)
    states = scale .* randn(rng, Float32, d, T)
    StructuredTrajectory(states, d, T)
end

function init_trajectory(d::Int, T::Int; scale::Float32=0.01f0)
    init_trajectory(Random.default_rng(), d, T; scale)
end

"""
Initialize a trajectory where z₁ is derived from the problem context h_x,
and subsequent states are noisy perturbations.
"""
function init_trajectory_from_context(
    rng::AbstractRNG, h_x::AbstractVector{Float32}, T::Int;
    scale::Float32=0.01f0,
)
    d = length(h_x)
    states = scale .* randn(rng, Float32, d, T)
    states[:, 1] .= h_x
    StructuredTrajectory(states, d, T)
end

"""
Return the trajectory as a flat matrix (d × T) for use in energy computations.
"""
function trajectory_to_matrix(traj::StructuredTrajectory)
    traj.states
end

"""
Return the final latent state z_T (used by the decoder).
"""
function final_state(traj::StructuredTrajectory)
    traj.states[:, end]
end

# ── Trajectory Transition Network ────────────────────────────

"""
Optional learned transition model that predicts z_{t+1} from z_t and h_x.
Can serve as an initialization prior or regularizer.
"""
struct TrajectoryTransition
    net::Chain
end

Flux.@layer TrajectoryTransition

function TrajectoryTransition(d::Int; hidden_dim::Int=128)
    net = Chain(
        Dense(2d, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, d),
    )
    TrajectoryTransition(net)
end

"""
Predict next state given current state and context.
"""
function step_trajectory(trans::TrajectoryTransition, z_t::AbstractVector, h_x::AbstractVector)
    input = vcat(z_t, h_x)
    trans.net(input)
end

"""
Unroll the transition model to produce a full trajectory.
"""
function unroll(trans::TrajectoryTransition, h_x::AbstractVector{Float32}, T::Int)
    d = length(h_x)
    states = zeros(Float32, d, T)
    states[:, 1] .= h_x
    for t in 1:(T-1)
        states[:, t+1] .= step_trajectory(trans, states[:, t], h_x)
    end
    StructuredTrajectory(states, d, T)
end

end # module
