module Visualize

using Plots
using Statistics
using JSON3
using CSV
using DataFrames
using LinearAlgebra
using MultivariateStats

export plot_energy_vs_steps, plot_trajectory_2d, plot_energy_landscape,
       plot_training_curves, plot_comparison, load_metrics,
       plot_energy_vs_constraint_satisfaction, plot_error_vs_energy,
       plot_planning_trajectory_pca, plot_ablation_line, plot_results_table

# ── Metrics Loading ──────────────────────────────────────────

function load_metrics(log_dir::String)
    path = joinpath(log_dir, "metrics.json")
    isfile(path) || error("Metrics file not found: $path")
    raw = JSON3.read(read(path, String))
    result = Dict{String, Vector{Tuple{Int, Float64}}}()
    for (k, v) in raw
        result[String(k)] = [(Int(pair[1]), Float64(pair[2])) for pair in v]
    end
    result
end

# ── Energy vs Planning Steps ─────────────────────────────────

"""
Plot how energy decreases during latent planning optimization.
Shows the energy trace from the planner.
"""
function plot_energy_vs_steps(
    energy_trace::Vector{<:Real};
    title::String="Energy During Latent Planning",
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(
        1:length(energy_trace), energy_trace;
        xlabel="Planning Step", ylabel="Energy E(x, z)",
        title=title, label="Energy",
        linewidth=2, color=:steelblue,
        grid=true, legend=:topright,
        size=(800, 500),
    )
    hline!(p, [minimum(energy_trace)]; label="Min Energy",
        linestyle=:dash, color=:firebrick, linewidth=1)

    if save_path !== nothing
        savefig(p, save_path)
        @info "Saved: $save_path"
    end
    p
end

"""
Plot energy traces from multiple runs overlaid.
"""
function plot_energy_vs_steps(
    traces::Vector{Vector{Float64}};
    labels::Vector{String}=["Run $i" for i in 1:length(traces)],
    title::String="Energy During Latent Planning",
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(; xlabel="Planning Step", ylabel="Energy E(x, z)",
        title=title, grid=true, size=(800, 500))
    colors = [:steelblue, :firebrick, :seagreen, :darkorange, :purple]
    for (i, trace) in enumerate(traces)
        c = colors[mod1(i, length(colors))]
        plot!(p, 1:length(trace), trace; label=labels[i], linewidth=2, color=c)
    end
    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── 2D Trajectory Visualization ──────────────────────────────

"""
Visualize latent trajectory projected to 2D via the first two principal dimensions.
Shows how z evolves from z₁ to z_T.
"""
function plot_trajectory_2d(
    z::AbstractMatrix;  # (d, T)
    title::String="Latent Trajectory",
    save_path::Union{String, Nothing}=nothing,
)
    d, T = size(z)
    if d < 2
        error("Need at least 2 latent dimensions for 2D projection")
    end

    x = z[1, :]
    y = z[2, :]

    p = plot(x, y;
        xlabel="z₁", ylabel="z₂", title=title,
        linewidth=2, color=:steelblue, label="Trajectory",
        marker=:circle, markersize=4, markerstrokewidth=0,
        grid=true, size=(700, 700),
    )
    scatter!(p, [x[1]], [y[1]]; label="Start (z₁)",
        markersize=10, color=:seagreen, markerstrokewidth=2)
    scatter!(p, [x[end]], [y[end]]; label="End (z_T)",
        markersize=10, color=:firebrick, markerstrokewidth=2)

    for t in 1:(T-1)
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        quiver!(p, [x[t]], [y[t]]; quiver=([dx * 0.8], [dy * 0.8]),
            color=:gray, linewidth=0.5)
    end

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Energy Landscape ─────────────────────────────────────────

"""
Plot the energy landscape around a trajectory point in 2D.
Varies two latent dimensions while holding others fixed at z_T values.
"""
function plot_energy_landscape(
    energy_model, h_x::AbstractVector, z_T::AbstractVector;
    dims::Tuple{Int,Int}=(1, 2),
    range_size::Float64=2.0,
    n_grid::Int=50,
    title::String="Energy Landscape",
    save_path::Union{String, Nothing}=nothing,
)
    d1, d2 = dims
    center1, center2 = z_T[d1], z_T[d2]
    xs = range(center1 - range_size, center1 + range_size; length=n_grid)
    ys = range(center2 - range_size, center2 + range_size; length=n_grid)

    energies = zeros(n_grid, n_grid)
    z_probe = copy(z_T)

    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        z_probe[d1] = x
        z_probe[d2] = y
        z_mat = reshape(z_probe, :, 1)
        energies[j, i] = energy_model(h_x, z_mat)
    end

    p = contourf(collect(xs), collect(ys), energies;
        xlabel="z[$(d1)]", ylabel="z[$(d2)]",
        title=title, color=:viridis,
        size=(700, 600),
    )
    scatter!(p, [center1], [center2]; label="z_T",
        markersize=8, color=:red, markerstrokewidth=2)

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Training Curves ──────────────────────────────────────────

"""
Plot training and validation loss curves from logged metrics.
"""
function plot_training_curves(
    metrics::Dict;
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(; xlabel="Step", ylabel="Loss",
        title="Training Progress", grid=true, size=(900, 500))

    if haskey(metrics, "train/loss_ma")
        steps, vals = unzip_metrics(metrics["train/loss_ma"])
        plot!(p, steps, vals; label="Train Loss (MA)", linewidth=2, color=:steelblue)
    end

    if haskey(metrics, "val/loss")
        steps, vals = unzip_metrics(metrics["val/loss"])
        scatter!(p, steps, vals; label="Val Loss",
            markersize=6, color=:firebrick, markerstrokewidth=1)
    end

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Solution Accuracy Comparison ─────────────────────────────

"""
Bar chart comparing accuracy/performance across tasks or methods.
"""
function plot_comparison(
    names::Vector{String}, values::Vector{Float64};
    ylabel::String="Accuracy (%)",
    title::String="Task Comparison",
    save_path::Union{String, Nothing}=nothing,
)
    p = bar(names, values;
        ylabel=ylabel, title=title,
        color=:steelblue, legend=false,
        size=(700, 500), bar_width=0.6,
    )
    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Helpers ──────────────────────────────────────────────────

function unzip_metrics(pairs)
    steps = [p[1] for p in pairs]
    vals = [p[2] for p in pairs]
    steps, vals
end

# ── Cross-Task Summary ───────────────────────────────────────

"""
Grouped bar chart for comparing direct vs planner across multiple tasks.
`tasks`: vector of task names.
`direct_vals`, `planner_vals`: parallel vectors of metric values.
"""
function plot_method_comparison(
    tasks::Vector{String},
    direct_vals::Vector{Float64},
    planner_vals::Vector{Float64};
    ylabel::String="Metric (%)",
    title::String="Direct vs Planner Across Tasks",
    save_path::Union{String, Nothing}=nothing,
)
    n = length(tasks)
    xs = 1:n
    w = 0.35

    p = plot(; xlabel="Task", ylabel=ylabel, title=title,
        xticks=(xs, tasks), grid=true, size=(800, 500), legend=:topleft)
    bar!(p, xs .- w/2, direct_vals; bar_width=w, label="Direct", color=:steelblue)
    bar!(p, xs .+ w/2, planner_vals; bar_width=w, label="Planner", color=:firebrick)

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

"""
Plot accuracy or recall vs planner steps for ablation studies.
`steps_list`: vector of planner step counts.
`values`: parallel vector of metric values.
"""
function plot_metric_vs_planner_steps(
    steps_list::Vector{Int},
    values::Vector{Float64};
    ylabel::String="Metric",
    title::String="Metric vs Planner Steps",
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(steps_list, values;
        xlabel="Planner Steps", ylabel=ylabel, title=title,
        marker=:circle, markersize=6, linewidth=2,
        color=:steelblue, grid=true, size=(700, 500),
    )
    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Energy vs Constraint Satisfaction ─────────────────────────

"""
Dual-axis plot: energy decreasing and clause satisfaction increasing over planner steps.
For logic tasks -- shows the relationship between energy minimization and constraint solving.
"""
function plot_energy_vs_constraint_satisfaction(
    energy_traces::Vector{Vector{Float64}},
    satisfaction_traces::Vector{Vector{Float64}};
    labels::Vector{String}=["Problem $i" for i in 1:length(energy_traces)],
    title::String="Energy vs Constraint Satisfaction During Planning",
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(; xlabel="Planning Step", size=(900, 500), title=title, grid=true, legend=:topright)
    colors = [:steelblue, :firebrick, :seagreen, :darkorange, :purple]

    for (i, (etrace, strace)) in enumerate(zip(energy_traces, satisfaction_traces))
        c = colors[mod1(i, length(colors))]
        steps = 1:length(etrace)
        plot!(p, steps, etrace; label="Energy: $(labels[i])", linewidth=2, color=c,
            ylabel="Energy E(x, z)")
    end

    p2 = twinx(p)
    for (i, strace) in enumerate(satisfaction_traces)
        c = colors[mod1(i, length(colors))]
        steps = 1:length(strace)
        plot!(p2, steps, strace .* 100; label="SAT%: $(labels[i])",
            linewidth=2, linestyle=:dash, color=c, ylabel="Clause Satisfaction (%)")
    end

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Error vs Energy (Arithmetic) ─────────────────────────────

"""
Scatter plot: final energy vs numerical error for arithmetic problems.
Shows whether lower energy correlates with lower prediction error.
"""
function plot_error_vs_energy(
    energies::Vector{Float64},
    errors::Vector{Float64};
    title::String="Final Energy vs Prediction Error",
    save_path::Union{String, Nothing}=nothing,
)
    p = scatter(energies, errors;
        xlabel="Final Energy E(x, z*)", ylabel="Absolute Error",
        title=title, color=:steelblue, alpha=0.6,
        markersize=4, markerstrokewidth=0,
        grid=true, size=(700, 500), legend=false,
    )

    if length(energies) > 2
        r = cor(energies, errors)
        annotate!(p, [(minimum(energies) + 0.1 * (maximum(energies) - minimum(energies)),
                       maximum(errors) * 0.9,
                       text("r = $(round(r; digits=3))", 10, :left))])
    end

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── PCA Trajectory Visualization ─────────────────────────────

"""
PCA projection of multiple latent trajectories, colored by example.
Each trajectory is a (d, T) matrix; projects all steps into a shared 2D space.
"""
function plot_planning_trajectory_pca(
    trajectories::Vector{<:AbstractMatrix};
    labels::Vector{String}=["Traj $i" for i in 1:length(trajectories)],
    title::String="Latent Trajectories (PCA Projection)",
    save_path::Union{String, Nothing}=nothing,
)
    all_points = hcat([t for t in trajectories]...)  # (d, total_steps)
    if size(all_points, 1) < 2
        error("Need at least 2 latent dimensions for PCA")
    end

    pca_model = fit(PCA, Float64.(all_points); maxoutdim=2)
    projected = predict(pca_model, Float64.(all_points))  # (2, total_steps)

    p = plot(; xlabel="PC1", ylabel="PC2", title=title, grid=true,
        size=(700, 700), legend=:outertopright)
    colors = [:steelblue, :firebrick, :seagreen, :darkorange, :purple,
              :teal, :crimson, :olive, :indigo, :coral]

    offset = 0
    for (i, traj) in enumerate(trajectories)
        T = size(traj, 2)
        xs = projected[1, offset+1:offset+T]
        ys = projected[2, offset+1:offset+T]
        c = colors[mod1(i, length(colors))]

        plot!(p, xs, ys; label=labels[i], linewidth=2, color=c,
            marker=:circle, markersize=3, markerstrokewidth=0)
        scatter!(p, [xs[1]], [ys[1]]; label="", markersize=7, color=c, markershape=:diamond)
        scatter!(p, [xs[end]], [ys[end]]; label="", markersize=7, color=c, markershape=:star5)
        offset += T
    end

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Ablation Line Plot ───────────────────────────────────────

"""
Line plot for ablation studies: shows direct vs planner metric across configs.
`config_vals`: x-axis values (e.g. planner steps, trajectory lengths).
"""
function plot_ablation_line(
    config_vals::Vector{<:Real},
    direct_vals::Vector{Float64},
    planner_vals::Vector{Float64};
    xlabel::String="Config Value",
    ylabel::String="Metric (%)",
    title::String="Ablation Study",
    save_path::Union{String, Nothing}=nothing,
)
    p = plot(; xlabel=xlabel, ylabel=ylabel, title=title, grid=true,
        size=(800, 500), legend=:topleft)

    plot!(p, config_vals, direct_vals;
        label="Direct", linewidth=2, color=:steelblue,
        marker=:circle, markersize=6)
    plot!(p, config_vals, planner_vals;
        label="Planner", linewidth=2, color=:firebrick,
        marker=:square, markersize=6)

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

# ── Results Summary Table ────────────────────────────────────

"""
Render a summary table as a bar chart with annotations.
`rows`: vector of NamedTuples with :label, :direct, :planner, :baseline fields.
"""
function plot_results_table(
    rows::Vector{<:NamedTuple};
    ylabel::String="Performance",
    title::String="EBRM Results Summary",
    save_path::Union{String, Nothing}=nothing,
)
    n = length(rows)
    labels = [r.label for r in rows]
    xs = 1:n
    w = 0.25

    p = plot(; xlabel="Task", ylabel=ylabel, title=title,
        xticks=(xs, labels), grid=true, size=(900, 500), legend=:topleft)

    baseline_vals = [get(r, :baseline, NaN) for r in rows]
    direct_vals = [r.direct for r in rows]
    planner_vals = [r.planner for r in rows]

    has_baseline = !all(isnan, baseline_vals)
    if has_baseline
        bar!(p, xs .- w, baseline_vals; bar_width=w, label="Baseline", color=:gray)
    end
    offset = has_baseline ? 0.0 : -w/2
    bar!(p, xs .+ offset, direct_vals; bar_width=w, label="EBRM Direct", color=:steelblue)
    bar!(p, xs .+ offset .+ w, planner_vals; bar_width=w, label="EBRM Planner", color=:firebrick)

    if save_path !== nothing
        savefig(p, save_path)
    end
    p
end

end # module
