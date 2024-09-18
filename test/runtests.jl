using RandomFeatureMaps
using Test

using BatchedTransformations

@testset "RandomFeatureMaps.jl" begin

    @testset "RandomFourierFeatures" begin
        rff = RandomFourierFeatures(10 => 20, 0.1f0)
        x = randn(Float32, 10, 4)
        @test rff(x) |> size == (20, 4)
        @test rff(reshape(x, 10, 2, 2)) |> size == (20, 2, 2)
    end

    @testset "RandomOrientationFeatures" begin
        dim = 10
        n = 8
        k = 5
        rof = RandomOrientationFeatures(dim, 0.1f0)
        rigid = rand_rigid(Float32, (n,k))
        @test rof(rigid, pairdim=1) |> size == (dim, n, n, k)
        @test rof((rand(3,3,n,k), rand(3,1,n,k)), pairdim=1) |> size == (dim, n, n, k)
        @test rof(rigid) == rof(rigid, rigid)
    end

    include("GraphNeuralNetworksExt.jl")

end
