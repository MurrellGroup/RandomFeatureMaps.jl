module RandomFeatureMaps

export RandomFourierFeatures
export RandomOrientationFeatures

using NNlib: batched_mul
using Functors: @functor
import Optimisers

"""
    RandomFourierFeatures(n => m, σ)

Maps `n`-dimensional data and projects it to `m`-dimensional random fourier features.

## Example

```jldoctest
julia> rff = RandomFourierFeatures(2 => 4, 1.0); # maps 2D data to 4D

julia> rff(rand(2, 3)) |> size # 3 samples
(4, 3)
```
"""
struct RandomFourierFeatures{T <: Real, A <: AbstractMatrix{T}}
    W::A
end

@functor RandomFourierFeatures
Optimisers.trainable(::RandomFourierFeatures) = (;)  # no trainable parameters

RandomFourierFeatures(dims::Pair{<:Integer, <:Integer}, σ::Real) =
    RandomFourierFeatures(dims, float(σ))

# d1: input dimension, d2: output dimension (d1 => d2)
function RandomFourierFeatures((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    iseven(d2) || throw(ArgumentError("dimension must be even"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomFourierFeatures(randn(typeof(σ), d1, d2 ÷ 2) * σ * oftype(σ, 2π))
end

function (rff::RandomFourierFeatures{T})(X::AbstractMatrix{T}) where T <: Real
    WtX = rff.W'X
    return [cos.(WtX); sin.(WtX)]
end

function (rff::RandomFourierFeatures{T})(X::AbstractArray{T}) where T <: Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rff(X′)
    Y = reshape(Y′, :, size(X)[2:end]...)
    return Y
end

"""
    RandomOrientationFeatures(m, σ)

Projects rigid transformations them to `m` features.
These will be the pairwise distances between points.
"""
struct RandomOrientationFeatures{A <: AbstractArray{<:Real}}
    FA::A
    FB::A
end

@functor RandomOrientationFeatures
Optimisers.trainable(::RandomOrientationFeatures) = (;)  # no trainable parameters

# should it just have a single array? such that pairwise distances require two of these
function RandomOrientationFeatures(dim::Integer, σ::AbstractFloat)
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomOrientationFeatures(randn(typeof(σ), 3, dim, 1) * σ, randn(typeof(σ), 3, dim, 1) * σ)
end

### For non-graph version, with batch dim
function transform_rigid(x::AbstractArray{T}, R::AbstractArray{T}, t::AbstractArray{T}) where T
    x′ = reshape(x, 3, size(x, 2), :)
    R′ = reshape(R, 3, 3, :)
    t′ = reshape(t, 3, 1, :)
    y′ = batched_mul(R′, x′) .+ t′
    y = reshape(y′, 3, size(x, 2), size(R)[3:end]...)
    return y
end
 
function (rof::RandomOrientationFeatures)(rigid::Tuple{AbstractArray, AbstractArray})
    dim = size(rof.FA, 2)
    Nr, batch... = size(rigid[1])[3:end]
    p1 = reshape(transform_rigid(rof.FA, rigid...), 3, dim, Nr, batch...)
    p2 = reshape(transform_rigid(rof.FB, rigid...), 3, dim, Nr, batch...)
    return dropdims(sqrt.(sum(abs2,
        reshape(p1, 3, dim, Nr, 1, batch...) .-
        reshape(p2, 3, dim, 1, Nr, batch...),
        dims=1)), dims=1)
end

### TODO: GRAPH version - see https://github.com/MurrellGroup/RandomFeatures.jl/blob/main/src/RandomFeatures.jl#L91

end
