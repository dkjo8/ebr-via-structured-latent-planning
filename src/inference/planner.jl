module Planner

using Flux
using Zygote
using Random
using Statistics

export LatentPlanner, optimize_latent, optimize_latent!, PlannerConfig,
       plan, multi_restart_plan

# ── Configuration ────────────────────────────────────────────

struct PlannerConfig
    steps::Int
    lr::Float64
    langevin_noise::Float64
    use_langevin::Bool
    clip_grad::Float64
    early_stop_tol::Float64
end

function PlannerConfig(;
    steps::Int=50,
    lr::Float64=0.01,
    langevin_noise::Float64=0.005,
    use_langevin::Bool=true,
    clip_grad::Float64=1.0,
    early_stop_tol::Float64=1e-6,
)
    PlannerConfig(steps, lr, langevin_noise, use_langevin, clip_grad, early_stop_tol)
end

# ── Latent Planner ───────────────────────────────────────────

struct LatentPlanner
    config::PlannerConfig
end

LatentPlanner(; kwargs...) = LatentPlanner(PlannerConfig(; kwargs...))

"""
    optimize_latent(energy_model, h_x, z_init, config) → (z_opt, energy_trace)

Optimize the latent trajectory z to minimize E(h_x, z) via gradient descent
(optionally with Langevin noise for exploration).

Returns the optimized trajectory matrix and a vector of energy values per step.
"""
function optimize_latent(
    energy_model, h_x::AbstractVector, z_init::AbstractMatrix;
    config::PlannerConfig=PlannerConfig(),
)
    z = copy(z_init)
    energy_trace = Float64[]
    prev_energy = Inf

    for i in 1:config.steps
        e_val, grads = Zygote.withgradient(z) do z_
            energy_model(h_x, z_)
        end

        push!(energy_trace, e_val)

        gz = grads[1]
        if gz === nothing
            @warn "No gradient at step $i"
            break
        end

        grad_norm = sqrt(sum(abs2, gz))
        if grad_norm > config.clip_grad
            gz = gz .* (config.clip_grad / grad_norm)
        end

        z .= z .- Float32(config.lr) .* gz

        if config.use_langevin
            noise_scale = Float32(config.langevin_noise * sqrt(2.0 * config.lr))
            noise = similar(z)
            noise .= randn!(noise)
            z .+= noise_scale .* noise
        end

        if abs(prev_energy - e_val) < config.early_stop_tol && i > 5
            break
        end
        prev_energy = e_val
    end

    z, energy_trace
end

"""
In-place version that modifies the StructuredTrajectory.
"""
function optimize_latent!(
    energy_model, h_x::AbstractVector, traj;
    config::PlannerConfig=PlannerConfig(),
)
    z_opt, trace = optimize_latent(energy_model, h_x, traj.states; config)
    traj.states .= z_opt
    traj, trace
end

"""
    plan(planner, energy_model, h_x, d, T; [rng]) → (z_opt, energy_trace)

Full planning pipeline: initialize a random trajectory, then optimize it.
"""
function plan(
    planner::LatentPlanner, energy_model, h_x::AbstractVector,
    d::Int, T::Int;
    rng::AbstractRNG=Random.default_rng(),
)
    z_init = 0.01f0 .* randn(rng, Float32, d, T)
    z_init[:, 1] .= h_x[1:min(d, length(h_x))]
    optimize_latent(energy_model, h_x, z_init; config=planner.config)
end

"""
    multi_restart_plan(planner, energy_model, h_x, d, T; n_restarts=5) → best (z, trace)

Run planning from multiple initializations and return the lowest-energy result.
"""
function multi_restart_plan(
    planner::LatentPlanner, energy_model, h_x::AbstractVector,
    d::Int, T::Int;
    n_restarts::Int=5,
    rng::AbstractRNG=Random.default_rng(),
)
    best_z = nothing
    best_trace = Float64[]
    best_energy = Inf

    for _ in 1:n_restarts
        z, trace = plan(planner, energy_model, h_x, d, T; rng)
        final_e = isempty(trace) ? Inf : last(trace)
        if final_e < best_energy
            best_energy = final_e
            best_z = z
            best_trace = trace
        end
    end

    best_z, best_trace
end

end # module
