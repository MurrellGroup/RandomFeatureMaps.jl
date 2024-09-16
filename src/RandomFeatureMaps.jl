module RandomFeatureMaps

export RandomFourierFeatures
export RandomOrientationFeatures
export rand_rigid, construct_rigid

using Flux: @functor, Optimisers, unsqueeze

using BatchedTransformations

"""
    RandomFourierFeatures(n => m, σ)

Maps `n`-dimensional data and projects it to `m`-dimensional random fourier features.

## Examples

```jldoctest
julia> rff = RandomFourierFeatures(2 => 4, 1.0); # maps 2D data to 4D

julia> rff(rand(2, 3)) |> size # 3 samples
(4, 3)

julia> rff(rand(2, 3, 5)) |> size # extra batch dim
(4, 3, 5)
```
"""
struct RandomFourierFeatures{T<:Real,A<:AbstractMatrix{T}}
    W::A
end

@functor RandomFourierFeatures
Optimisers.trainable(::RandomFourierFeatures) = (;)  # no trainable parameters

RandomFourierFeatures(dims::Pair{<:Integer, <:Integer}, σ::Real) = RandomFourierFeatures(dims, float(σ))

# d1: input dimension, d2: output dimension (d1 => d2)
function RandomFourierFeatures((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    iseven(d2) || throw(ArgumentError("dimension must be even"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomFourierFeatures(randn(typeof(σ), d1, d2 ÷ 2) * σ * oftype(σ, 2π))
end

function (rff::RandomFourierFeatures{T})(X::AbstractMatrix{T}) where T<:Real
    WtX = rff.W'X
    return [cos.(WtX); sin.(WtX)]
end

function (rff::RandomFourierFeatures{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rff(X′)
    Y = reshape(Y′, :, size(X)[2:end]...)
    return Y
end

rand_rigid(T::Type, batch_size::Dims) = rand(T, Rigid, 3, batch_size)

function construct_rigid(R::AbstractArray, t::AbstractArray)
    batch_size = size(R)[3:end]
    t = reshape(t, 3, 1, batch_size...)
    Translation(t) ∘ Rotation(R)
end

"""
    RandomOrientationFeatures

Can be called on rigid transformations to create pairwise maps of random orientation features.
This type has no trainable parameters.

## Examples

```jldoctest
julia> rof = RandomOrientationFeatures(4, 0.1);

julia> rigid = (randn(3, 3, 2), randn(3, 1, 2)); # cba to make it orthonormal and whatevs

julia> rof(rigid) |> size
(4, 2, 2)
```
"""
struct RandomOrientationFeatures{T<:Real,A<:AbstractMatrix{T}}
    FA::A
    FB::A
end

@functor RandomOrientationFeatures
Optimisers.trainable(::RandomOrientationFeatures) = (;)  # no trainable parameters

"""
    RandomOrientationFeatures(m, σ)

Creates a `RandomOrientationFeatures` instance, mapping to `m` features.
"""
function RandomOrientationFeatures(dim::Integer, σ::AbstractFloat)
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomOrientationFeatures(randn(typeof(σ), 3, dim) * σ, randn(typeof(σ), 3, dim) * σ)
end

function (rof::RandomOrientationFeatures)(rigid::Rigid)
    points1 = rigid * rof.FA
    points2 = rigid * rof.FB
    diffs = unsqueeze(points1, dims=4) .- unsqueeze(points2, dims=3)
    return dropdims(sqrt.(sum(abs2, diffs; dims=1)); dims=1)
end

(rof::RandomOrientationFeatures)((R, t)::Tuple{AbstractArray,AbstractArray}, args...) = rof(construct_rigid(R, t), args...) 

end
