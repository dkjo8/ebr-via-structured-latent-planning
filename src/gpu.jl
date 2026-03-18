module GPU

using Flux

export use_gpu, to_device, to_cpu, device_randn

const _use_gpu = Ref(false)

function __init__()
    _use_gpu[] = false
    try
        @eval using CUDA
        if CUDA.functional()
            _use_gpu[] = true
            @info "CUDA available: $(CUDA.device())"
        else
            @info "CUDA.jl loaded but not functional, using CPU"
        end
    catch
        @info "CUDA.jl not available, using CPU"
    end
end

use_gpu()::Bool = _use_gpu[]

function to_device(x)
    use_gpu() ? gpu(x) : x
end

function to_cpu(x)
    cpu(x)
end

"""
    device_randn(Float32, dims...) → array on current device
"""
function device_randn(::Type{T}, dims...) where T
    x = randn(T, dims...)
    use_gpu() ? gpu(x) : x
end

function device_randn(rng, ::Type{T}, dims...) where T
    x = randn(rng, T, dims...)
    use_gpu() ? gpu(x) : x
end

end # module
