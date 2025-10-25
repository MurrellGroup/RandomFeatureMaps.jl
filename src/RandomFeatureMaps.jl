module RandomFeatureMaps

export RandomFourierFeatures
export RandomOrientationFeatures
export RandomTriangleFeatures
export TrainableRBF
export rand_rigid, get_rigid, trianglecos, trianglesin

using Flux
using BatchedTransformations
using ChainRulesCore

sumdrop(f, A::AbstractArray; dims) = dropdims(sum(f, A; dims); dims)
norms(A::AbstractArray; dims) = sqrt.(sumdrop(abs2, A; dims))

trianglecos(x::T) where T = T(2abs(mod(x, 2)-1)-1)
dtrianglecos(x::T) where T = mod(x, 2) < 1 ? T(-2) : T(2)
trianglesin(x::T) where T = trianglecos(x-0.5)

trianglecos(x::AbstractArray{T}) where T = @. 2abs(mod(x, 2)-1)-1
trianglesin(x::AbstractArray{T}) where T = trianglecos(x .- T(0.5))

function ChainRulesCore.rrule(::typeof(trianglecos), x::AbstractArray{T}) where {T}
    y = trianglecos(x)
    function trianglecos_pullback(ȳ)
        x̄ = ȳ .* dtrianglecos.(x)
        return NoTangent(), x̄
    end
    return y, trianglecos_pullback
end

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
struct RandomTriangleFeatures{T<:Real,A<:AbstractMatrix{T}}
    W::A
end

Flux.@layer RandomTriangleFeatures trainable=()

RandomTriangleFeatures(dims::Pair{<:Integer, <:Integer}, σ::Real) = RandomTriangleFeatures(dims, float(σ))

# d1: input dimension, d2: output dimension (d1 => d2)
function RandomTriangleFeatures((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    iseven(d2) || throw(ArgumentError("dimension must be even"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomTriangleFeatures(randn(typeof(σ), d1, d2 ÷ 2) * σ * oftype(σ, 2π))
end

function (rff::RandomTriangleFeatures{T})(X::AbstractMatrix{T}) where T<:Real
    Y = rff.W'X
    return [trianglecos(Y); trianglesin(Y)]
end

function (rff::RandomTriangleFeatures{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rff(X′)
    Y = reshape(Y′, :, size(X)[2:end]...)
    return Y
end

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

Flux.@layer RandomFourierFeatures trainable=()

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

julia> rigid = rand_rigid(Float32, (4, 3));

julia> rof(rigid, rigid) |> size
(10, 4, 3)

julia> rigid1, rigid2 = rand_rigid(Float32, (4, 2)), rand_rigid(Float32, (3, 2));

julia> rof(rigid1, rigid2; pairdim=1) |> size
(10, 4, 3, 2)

julia> using GraphNeuralNetworks

julia> graph = GNNGraph(Bool[1 0; 1 1], graph_type=:dense)
GNNGraph:
  num_nodes: 2
  num_edges: 3

julia> rigid = rand_rigid(Float32, (2,));

julia> rof(rigid, graph) |> size
(10, 3)
```
"""
struct RandomOrientationFeatures{A<:AbstractArray{<:Real}}
    FA::A
    FB::A
end

Flux.@layer RandomOrientationFeatures trainable=()

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
    t = reshape(t, 3, 1, size(R)[3:end]...)
    Translation(t) ∘ Rotation(R)
end

(rof::RandomOrientationFeatures)(T1::Tuple, T2::Tuple, args...; kwargs...) =
    rof(get_rigid(T1...), get_rigid(T2...), args...; kwargs...)




"""
    TrainableRBF(n => m, [σ])

Maps `n`-dimensional data to `m` Gaussian radial basis responses with trainable
centers and isotropic radii per basis.

The optional `σ` controls the initialization scale and element type.

Examples

```jldoctest
julia> rbf = TrainableRBF(2 => 4, 1.0); # 4 bases in 2D

julia> rbf(rand(2, 3)) |> size # 3 samples
(4, 3)

julia> rbf(rand(2, 3, 5)) |> size # extra batch dim
(4, 3, 5)
```
"""
struct TrainableRBF{T<:Real, A<:AbstractMatrix{T}, V<:AbstractVector{T}}
    centers::A  # (n, m)
    radii::V    # (m,) isotropic per basis
end

Flux.@layer TrainableRBF trainable=(centers, radii)

TrainableRBF(dims::Pair{<:Integer, <:Integer}, σ::Real=1.0) = TrainableRBF(dims, float(σ))

function TrainableRBF((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    isfinite(σ) || throw(ArgumentError("scale must be finite"))
    centers = randn(typeof(σ), d1, d2) * σ
    radii = fill(typeof(σ)(1), d2)
    return TrainableRBF(centers, radii)
end

function (rbf::TrainableRBF{T})(X::AbstractMatrix{T}) where T<:Real
    C = rbf.centers
    R = rbf.radii
    X2 = sum(abs2, X; dims = 1)                 # (1, N)
    C2 = sum(abs2, C; dims = 1)                 # (1, M)
    D2 = (-2 .* (C' * X)) .+ C2' .+ X2          # (M, N)
    σ = softplus.(R) .+ T(1e-6)
    denom = 2 .* reshape(σ .^ 2, :, 1)          # (M, 1)
    return exp.(-D2 ./ denom)                   # (M, N)
end

function (rbf::TrainableRBF{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rbf(X′)
    return reshape(Y′, :, size(X)[2:end]...)
end

end
