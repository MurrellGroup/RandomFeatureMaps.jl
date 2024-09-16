using GraphNeuralNetworks

@testset "GraphNeuralNetworksExt.jl" begin

    @testset "Equivalence" begin
        m = 10
        n = 8
        rigid = rand_rigid(Float32, (n,))
        g = ones(Bool, n, n)
        graph = GNNGraph(g, graph_type=:dense)
        rof = RandomOrientationFeatures(m, 0.1f0)
        @test rof(rigid, graph) == reshape(rof(rigid), m, :)
        
        @test rof((rand(3,3,n), rand(3,1,n)), graph) |> size == (m, n^2) # deprecated
    end

    @testset "Random graph" begin
        m = 10
        n = 8
        rigid = rand_rigid(Float32, (n,))
        g = rand(Bool, n, n)
        graph = GNNGraph(g, graph_type=:dense)
        rof = RandomOrientationFeatures(m, 0.1f0)
        @test size(rof(rigid, graph), 2) == count(g)
        @test rof(rigid, graph) == reshape(rof(rigid), m, :)[:,findall(vec(g))]
    end

end