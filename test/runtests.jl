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
        rof = RandomOrientationFeatures(10, 0.1f0)
        @test rof(rand_rigid(Float32, (4,2))) |> size == (10, 4, 4, 2)
        @test rof((rand(3,3,4,2), rand(3,1,4,2))) |> size == (10, 4, 4, 2) # deprecated
    end

    include("GraphNeuralNetworksExt.jl")

end
