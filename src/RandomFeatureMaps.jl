module RandomFeatureMaps

export RandomFourierFeatures
export RandomOrientationFeatures
export pairwiserof
export rand_rigid, get_rigid

using Functors: @functor
import Optimisers

using BatchedTransformations

sumdrop(f, A::AbstractArray; dims) = dropdims(sum(f, A; dims); dims)
norms(A::AbstractArray; dims) = sqrt.(sumdrop(abs2, A; dims))

"""
    RandomFourierFeatures(n => m, σ)

Maps `n`-dimensional data and projects it to `m`-dimensional random fourier features.

This type has no trainable parameters.

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
    Y = rff.W'X
    return [cos.(Y); sin.(Y)]
end

function (rff::RandomFourierFeatures{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rff(X′)
    Y = reshape(Y′, :, size(X)[2:end]...)
    return Y
end


"""
    RandomOrientationFeatures

Holds two random matrices which are used to embed rigid transformations.

This type has no trainable parameters.

## Methods

- `(::RandomOrientationFeatures)(rigid1, rigid2)`: returns the distances between the corresponding
rigid transformations, embedded using the two random matrices of the random orientation features.

- `(::RandomOrientationFeatures)(rigid1, rigid2; dims::Int)`: unsqueezes batch dimension `dim+1`
of `rigid1` and `dim` of `rigid2` to broadcast the `rof` call and produce a pairwise map.

- `(::RandomOrientationFeatures)(rigid1, rigid2, graph::GraphNeuralNetworks.GNNGraph)`: similar to
the first method, but takes two sets rigid transformations of equal size and unrolls a graph to
get the pairs of rigid transformations. Equivalent to the second method (with broadcasted dimensions
flattened) when the graph is complete.

Each of these have single rigid argument methods for when `rigid1 == rigid2`, i.e. `rof(rigid)`

## Examples

```jldoctest
julia> rof = RandomOrientationFeatures(10, 0.1f0);

julia> rigid = rand_rigid(Float32, (2, 3));

julia> rof(rigid, rigid) |> size
(10, 4, 3)

julia> rigid1, rigid2 = rand_rigid(Float32, (4, 2)), rand_rigid(Float32, (3, 2));

julia> rof(rigid1, rigid2; dims=1) |> size
(10, 4, 3, 2)

julia> using GraphNeuralNetworks

julia> graph = GNNGraph(rand(Bool, 4, 4), graph_type=:dense)

julia> rigid = rand_rigid(Float32, (4,));

julia> rof(rigid, graph) |> size
```
"""
struct RandomOrientationFeatures{A<:AbstractArray{<:Real}}
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

_rof(rof::RandomOrientationFeatures, T1::Rigid, T2::Rigid) = norms(T1(rof.FA) .- T2(rof.FB); dims=1)

function (rof::RandomOrientationFeatures)(T1::Rigid, T2::Rigid; pairdim::Union{Nothing,Int}=nothing)
    if pairdim isa Int
        T1, T2 = batchunsqueeze(T1, dims=pairdim+1), batchunsqueeze(T2, dims=pairdim)
    end
    _rof(rof, T1, T2)
end

(rof::RandomOrientationFeatures)(T; kwargs...) = rof(T, T; kwargs...)


rand_rigid(T::Type, batch_size::Dims) = rand(T, Rigid, 3, batch_size)

"""
    get_rigid(R::AbstractArray, t::AbstractArray)

Converts a rotation `R` and translation `t` to a `BatchedTransformations.Rigid`, designed to
handle batch dimensions.

The transformation gets applied according to `NNlib.batched_mul(R,  x) .+ t`
"""
function get_rigid(R::AbstractArray, t::AbstractArray)
    batch_size = size(R)[3:end]
    t = reshape(t, 3, 1, batch_size...)
    Translation(t) ∘ Rotation(R)
end

(rof::RandomOrientationFeatures)(T1::Tuple, T2::Tuple, args...; kwargs...) =
    rof(get_rigid(T1...), get_rigid(T2...), args...; kwargs...)

end
