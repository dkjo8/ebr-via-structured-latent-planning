module Utils

using TOML
using Random
using Statistics
using Dates
using JSON3
using Flux

export load_config, set_seed!, MovingAverage, update!, value,
       MetricsLogger, log_metric!, log_metrics!, save_metrics,
       log_config!, checkpoint_model, load_checkpoint,
       write_temp_config,
       @timed_block

# ── Configuration ────────────────────────────────────────────

function load_config(path::String="config.toml")
    isfile(path) || error("Config file not found: $path")
    TOML.parsefile(path)
end

function set_seed!(seed::Int)
    Random.seed!(seed)
end

"""
Write a config dict to a temporary TOML file. Returns the file path.
`tag` should be unique per invocation (e.g. \"sweep_graph_1\", \"ablation_A_logic_3\").
"""
function write_temp_config(cfg::Dict, tag::String)
    path = joinpath(tempdir(), "ebrm_config_$(tag).toml")
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

# ── Moving Average ───────────────────────────────────────────

mutable struct MovingAverage
    window::Int
    buffer::Vector{Float64}
    idx::Int
    count::Int
end

MovingAverage(window::Int=100) = MovingAverage(window, zeros(window), 0, 0)

function update!(ma::MovingAverage, val::Real)
    ma.idx = mod1(ma.idx + 1, ma.window)
    ma.buffer[ma.idx] = Float64(val)
    ma.count = min(ma.count + 1, ma.window)
    ma
end

function value(ma::MovingAverage)
    ma.count == 0 && return 0.0
    mean(@view ma.buffer[1:ma.count])
end

# ── Metrics Logger ───────────────────────────────────────────

mutable struct MetricsLogger
    run_name::String
    log_dir::String
    history::Dict{String, Vector{Tuple{Int, Float64}}}
    step::Int
end

function MetricsLogger(; run_name::String="run", log_dir::String="runs")
    ts = Dates.format(now(), "yyyymmdd_HHMMss")
    name = "$(run_name)_$(ts)"
    dir = joinpath(log_dir, name)
    mkpath(dir)
    MetricsLogger(name, dir, Dict{String, Vector{Tuple{Int, Float64}}}(), 0)
end

function log_metric!(logger::MetricsLogger, key::String, val::Real; step::Int=logger.step)
    vec = get!(logger.history, key, Tuple{Int, Float64}[])
    push!(vec, (step, Float64(val)))
end

function log_metrics!(logger::MetricsLogger, pairs::Dict{String, <:Real}; step::Int=logger.step)
    for (k, v) in pairs
        log_metric!(logger, k, v; step)
    end
end

function save_metrics(logger::MetricsLogger)
    path = joinpath(logger.log_dir, "metrics.json")
    open(path, "w") do io
        JSON3.write(io, logger.history)
    end
    path
end

"""
Save a config dict alongside the metrics run for reproducibility.
"""
function log_config!(logger::MetricsLogger, cfg::Dict)
    path = joinpath(logger.log_dir, "config.json")
    open(path, "w") do io
        JSON3.write(io, cfg)
    end
    path
end

# ── Checkpointing ────────────────────────────────────────────

function checkpoint_model(model, optimizer_state, epoch::Int, logger::MetricsLogger)
    path = joinpath(logger.log_dir, "checkpoint_epoch$(lpad(epoch, 4, '0')).json")
    model_state = Flux.state(model)
    open(path, "w") do io
        JSON3.write(io, Dict("epoch" => epoch, "model_state" => model_state))
    end
    path
end

function load_checkpoint(path::String)
    isfile(path) || error("Checkpoint not found: $path")
    JSON3.read(read(path, String))
end

# ── Timing ───────────────────────────────────────────────────

macro timed_block(name, expr)
    quote
        local t0 = time()
        local result = $(esc(expr))
        local elapsed = time() - t0
        @info "$($(esc(name))): $(round(elapsed; digits=3))s"
        result
    end
end

end # module
