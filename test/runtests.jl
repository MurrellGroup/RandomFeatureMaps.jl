using RandomFeatureMaps
using Test

@testset "RandomFeatureMaps.jl" begin
    rff = RandomFourierFeatures(10 => 20, 0.1f0)
    x = randn(Float32, 10, 4)
    @test rff(x) |> size == (20, 4)
    @test rff(reshape(x, 10, 2, 2)) |> size == (20, 2, 2)

    rof = RandomOrientationFeatures(10, 0.1f0)
    @test rof((randn(Float32, 3, 3, 4, 2), randn(Float32, 3, 1, 4, 2))) |> size == (10, 4, 4, 2)
end
