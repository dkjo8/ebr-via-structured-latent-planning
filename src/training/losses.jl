module Losses

using Flux
using Statistics

export contrastive_energy_loss, score_matching_loss, decoder_loss,
       trajectory_smoothness_loss, combined_loss

# ── Contrastive Energy Loss ──────────────────────────────────

"""
    contrastive_energy_loss(energy_model, h_x, z_pos, z_neg; margin=1.0)

Push energy of positive (optimized) trajectories below negative (random)
trajectories by at least `margin`.

L = max(0, E(x, z⁺) - E(x, z⁻) + margin)
"""
function contrastive_energy_loss(
    energy_model, h_x::AbstractVector,
    z_pos::AbstractMatrix, z_neg::AbstractMatrix;
    margin::Float32=1.0f0,
)
    e_pos = energy_model(h_x, z_pos)
    e_neg = energy_model(h_x, z_neg)
    max(0f0, e_pos - e_neg + margin)
end

"""
Batched version: z_pos_batch and z_neg_batch are vectors of trajectory matrices.
"""
function contrastive_energy_loss(
    energy_model, h_x_batch::Vector,
    z_pos_batch::Vector, z_neg_batch::Vector;
    margin::Float32=1.0f0,
)
    n = length(h_x_batch)
    total = 0f0
    for i in 1:n
        total += contrastive_energy_loss(
            energy_model, h_x_batch[i], z_pos_batch[i], z_neg_batch[i]; margin
        )
    end
    total / n
end

# ── Score Matching Loss ──────────────────────────────────────

"""
    score_matching_loss(energy_model, h_x, z_star; noise_scale=0.01)

Denoising score matching: perturb z* with noise, then train the energy
gradient to point back toward z*.

This avoids needing explicit negative samples.
"""
function score_matching_loss(
    energy_model, h_x::AbstractVector, z_star::AbstractMatrix;
    noise_scale::Float32=0.01f0,
)
    noise = noise_scale .* randn(Float32, size(z_star))
    z_noisy = z_star .+ noise
    target_score = -noise ./ (noise_scale^2)

    e_val, grad = Flux.withgradient(z_noisy) do z_
        energy_model(h_x, z_)
    end

    gz = grad[1]
    gz === nothing && return 0f0

    mean((gz .- target_score).^2)
end

# ── Decoder Loss ─────────────────────────────────────────────

"""
    decoder_loss(decoder, z_T, target; loss_fn=Flux.binarycrossentropy)

Supervised loss on the decoder output given the final latent state.
Default is binary cross-entropy (decoder outputs should be in [0, 1]).
"""
function decoder_loss(decoder, z_T::AbstractVector, target; loss_fn=Flux.binarycrossentropy)
    prediction = decoder(z_T)
    loss_fn(prediction, target)
end

# ── Trajectory Smoothness ────────────────────────────────────

"""
Regularizer penalizing large jumps between consecutive latent states.
"""
function trajectory_smoothness_loss(z::AbstractMatrix)
    size(z, 2) <= 1 && return 0f0
    diffs = z[:, 2:end] .- z[:, 1:end-1]
    mean(sum(diffs.^2; dims=1))
end

# ── Combined Loss ────────────────────────────────────────────

"""
    combined_loss(energy_model, decoder, h_x, z_pos, z_neg, z_T, target;
                  α_contrastive=1.0, α_decoder=1.0, α_smooth=0.1, margin=1.0)

Weighted combination of all loss terms.
"""
function combined_loss(
    energy_model, decoder, h_x, z_pos, z_neg, target;
    α_contrastive::Float32=1.0f0,
    α_decoder::Float32=1.0f0,
    α_smooth::Float32=0.1f0,
    margin::Float32=1.0f0,
)
    l_contrast = contrastive_energy_loss(energy_model, h_x, z_pos, z_neg; margin)
    z_T = z_pos[:, end]
    l_decode = decoder_loss(decoder, z_T, target)
    l_smooth = trajectory_smoothness_loss(z_pos)
    α_contrastive * l_contrast + α_decoder * l_decode + α_smooth * l_smooth
end

end # module
