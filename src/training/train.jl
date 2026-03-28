module Train

using Flux
using Zygote
using Random
using Statistics
using Dates

include("losses.jl")

include(joinpath(@__DIR__, "..", "models", "encoder.jl"))
include(joinpath(@__DIR__, "..", "models", "latent_trajectory.jl"))
include(joinpath(@__DIR__, "..", "models", "energy_network.jl"))
include(joinpath(@__DIR__, "..", "models", "decoder.jl"))
include(joinpath(@__DIR__, "..", "inference", "planner.jl"))
include(joinpath(@__DIR__, "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "gpu.jl"))

using .Losses
using .Encoder
using .LatentTrajectory
using .EnergyNetwork
using .Decoder
using .Planner
using .Utils
using .GPU

export TrainConfig, TrainState, EBRMSystem, train_epoch!, train!, evaluate, GPU

# ── Configuration ────────────────────────────────────────────

struct TrainConfig
    epochs::Int
    batch_size::Int
    lr::Float64
    weight_decay::Float64
    checkpoint_interval::Int
    α_contrastive::Float32
    α_decoder::Float32
    α_smooth::Float32
    margin::Float32
    planner_config::Planner.PlannerConfig
    decoder_loss_fn::Function
    patience::Int
    min_delta::Float64
    dual_path_decoder::Bool
end

function TrainConfig(cfg::Dict; decoder_loss_fn::Function=Flux.binarycrossentropy,
                     patience::Int=15, min_delta::Float64=1e-4)
    tc = cfg["training"]
    ic = cfg["inference"]
    TrainConfig(
        tc["epochs"],
        tc["batch_size"],
        tc["learning_rate"],
        tc["weight_decay"],
        tc["checkpoint_interval"],
        Float32(get(tc, "alpha_contrastive", 0.1)),
        Float32(get(tc, "alpha_decoder", 1.0)),
        Float32(get(tc, "alpha_smooth", 0.01)),
        1.0f0,
        Planner.PlannerConfig(;
            steps=ic["planner_steps"],
            lr=ic["planner_lr"],
            langevin_noise=ic["langevin_noise"],
            use_langevin=get(ic, "use_langevin", false),
            anchor_weight=get(ic, "anchor_weight", 0.0),
        ),
        decoder_loss_fn,
        patience,
        min_delta,
        get(tc, "dual_path_decoder", false),
    )
end

# ── Full System ──────────────────────────────────────────────

struct EBRMSystem
    encoder::Any
    energy_model::EnergyNetwork.EnergyModel
    decoder::Any
    latent_dim::Int
    trajectory_length::Int
end

function all_models(sys::EBRMSystem)
    (sys.encoder, sys.energy_model, sys.decoder)
end

mutable struct TrainState
    epoch::Int
    step::Int
    best_val_loss::Float64
    logger::Utils.MetricsLogger
    loss_ma::Utils.MovingAverage
end

function TrainState(; run_name="train")
    TrainState(0, 0, Inf, Utils.MetricsLogger(; run_name), Utils.MovingAverage(50))
end

# ── Training Loop ────────────────────────────────────────────

function train_epoch!(
    system::EBRMSystem,
    data,
    prepare_fn::Function,
    opt_state,
    config::TrainConfig,
    state::TrainState,
    rng::AbstractRNG,
)
    d = system.latent_dim
    T = system.trajectory_length
    total_loss = 0.0
    n_samples = 0
    t_epoch_start = time()

    opt_encdec, opt_energy = opt_state

    indices = randperm(rng, length(data))
    n_batches = ceil(Int, length(data) / config.batch_size)

    for batch_idx in 1:n_batches
        start_i = (batch_idx - 1) * config.batch_size + 1
        end_i = min(batch_idx * config.batch_size, length(data))
        batch_indices = indices[start_i:end_i]

        batch_loss = 0f0

        for idx in batch_indices
            sample = data[idx]
            features, target = prepare_fn(sample)
            features = GPU.to_device(features)
            target = GPU.to_device(target)

            # Step 1: Run planner to get z_pos (needed for dual-path decoder training).
            h_x_detached = system.encoder(features)
            z_init = 0.01f0 .* GPU.device_randn(rng, Float32, d, T)
            z_init[:, 1] .= h_x_detached
            z_pos, _ = Planner.optimize_latent(
                system.energy_model, h_x_detached, z_init;
                config=config.planner_config,
            )

            # Step 2: Train encoder + decoder (direct path, and optionally on z_T).
            l_dec, g_dec = Flux.withgradient(system.encoder, system.decoder) do enc, dec
                h = enc(features)
                l_direct = Losses.decoder_loss(dec, h, target; loss_fn=config.decoder_loss_fn)
                if config.dual_path_decoder
                    l_planned = Losses.decoder_loss(dec, z_pos[:, end], target; loss_fn=config.decoder_loss_fn)
                    0.5f0 * l_direct + 0.5f0 * l_planned
                else
                    l_direct
                end
            end
            opt_encdec, _ = Flux.update!(opt_encdec, (system.encoder, system.decoder), g_dec)

            # Step 3: Train energy model with contrastive loss.
            h_x = system.encoder(features)
            z_neg = z_pos .+ 0.5f0 .* GPU.device_randn(rng, Float32, d, T)

            l_energy, g_energy = Flux.withgradient(system.energy_model) do em
                l_con = Losses.contrastive_energy_loss(em, h_x, z_pos, z_neg; margin=config.margin)
                l_smooth = Losses.trajectory_smoothness_loss(z_pos)
                config.α_contrastive * l_con + config.α_smooth * l_smooth
            end
            opt_energy, _ = Flux.update!(opt_energy, (system.energy_model,), g_energy)

            batch_loss += l_dec + l_energy
            n_samples += 1
        end

        avg_batch_loss = batch_loss / length(batch_indices)
        Utils.update!(state.loss_ma, avg_batch_loss)
        state.step += 1

        Utils.log_metric!(state.logger, "train/loss", avg_batch_loss; step=state.step)
        Utils.log_metric!(state.logger, "train/loss_ma", Utils.value(state.loss_ma); step=state.step)

        total_loss += batch_loss
    end

    elapsed = time() - t_epoch_start
    samples_per_sec = n_samples / max(elapsed, 1e-6)
    Utils.log_metric!(state.logger, "train/epoch_seconds", elapsed; step=state.step)
    Utils.log_metric!(state.logger, "train/samples_per_sec", samples_per_sec; step=state.step)

    state.epoch += 1
    total_loss / max(n_samples, 1)
end

function evaluate(
    system::EBRMSystem,
    data,
    prepare_fn::Function,
    config::TrainConfig,
    rng::AbstractRNG,
)
    d = system.latent_dim
    T = system.trajectory_length
    total_loss = 0.0
    n = length(data)

    for i in 1:n
        sample = data[i]
        features, target = prepare_fn(sample)
        features = GPU.to_device(features)
        target = GPU.to_device(target)

        h_x = system.encoder(features)
        z_init = 0.01f0 .* GPU.device_randn(rng, Float32, d, T)
        z_init[:, 1] .= h_x
        z_opt, _ = Planner.optimize_latent(
            system.energy_model, h_x, z_init;
            config=config.planner_config,
        )

        z_T = z_opt[:, end]
        pred = system.decoder(z_T)
        total_loss += config.decoder_loss_fn(pred, target)
    end

    Dict("val/loss" => total_loss / n)
end

function train!(
    system::EBRMSystem,
    train_data,
    val_data,
    prepare_fn::Function,
    config::TrainConfig;
    seed::Int=42,
)
    rng = MersenneTwister(seed)
    state = TrainState(; run_name="ebrm")

    run_config = Dict(
        "latent_dim" => system.latent_dim,
        "trajectory_length" => system.trajectory_length,
        "epochs" => config.epochs,
        "batch_size" => config.batch_size,
        "lr" => config.lr,
        "alpha_contrastive" => config.α_contrastive,
        "alpha_decoder" => config.α_decoder,
        "alpha_smooth" => config.α_smooth,
        "planner_steps" => config.planner_config.steps,
        "planner_lr" => config.planner_config.lr,
        "anchor_weight" => config.planner_config.anchor_weight,
        "dual_path_decoder" => config.dual_path_decoder,
        "seed" => seed,
    )
    Utils.log_config!(state.logger, run_config)

    if GPU.use_gpu()
        @info "Moving models to GPU"
        system = EBRMSystem(
            GPU.to_device(system.encoder),
            GPU.to_device(system.energy_model),
            GPU.to_device(system.decoder),
            system.latent_dim,
            system.trajectory_length,
        )
    end

    opt_encdec = Flux.setup(Adam(config.lr), (system.encoder, system.decoder))
    opt_energy = Flux.setup(Adam(config.lr), (system.energy_model,))

    @info "Starting training" epochs=config.epochs lr=config.lr patience=config.patience device=(GPU.use_gpu() ? "GPU" : "CPU")

    epochs_no_improve = 0
    stopped_early = false

    for epoch in 1:config.epochs
        epoch_loss = train_epoch!(system, train_data, prepare_fn, (opt_encdec, opt_energy), config, state, rng)

        Utils.log_metric!(state.logger, "train/epoch_loss", epoch_loss; step=state.step)
        @info "Epoch $epoch/$(config.epochs)" loss=round(epoch_loss; digits=4) ma=round(Utils.value(state.loss_ma); digits=4)

        if epoch % config.checkpoint_interval == 0 || epoch == config.epochs
            val_metrics = evaluate(system, val_data, prepare_fn, config, rng)
            val_loss = val_metrics["val/loss"]
            Utils.log_metric!(state.logger, "val/loss", val_loss; step=state.step)
            @info "  Validation" val_loss=round(val_loss; digits=4)

            if val_loss < state.best_val_loss - config.min_delta
                state.best_val_loss = val_loss
                epochs_no_improve = 0
                Utils.checkpoint_model(all_models(system), nothing, epoch, state.logger)
                @info "  New best model saved"
            else
                epochs_no_improve += config.checkpoint_interval
            end

            if config.patience > 0 && epochs_no_improve >= config.patience
                @info "Early stopping at epoch $epoch (no improvement for $epochs_no_improve epochs)"
                Utils.log_metric!(state.logger, "train/early_stop_epoch", Float64(epoch); step=state.step)
                stopped_early = true
                break
            end
        end
    end

    if !stopped_early
        @info "Training completed all $(config.epochs) epochs"
    end

    Utils.save_metrics(state.logger)
    @info "Training complete. Logs saved to $(state.logger.log_dir)"
    state
end

end # module
