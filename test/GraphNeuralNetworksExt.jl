using GraphNeuralNetworks

@testset "GraphNeuralNetworksExt.jl" begin

    @testset "Complete graph equivalence" begin
        dim = 10
        n = 5
        rigid = rand_rigid(Float32, (n,))
        g = ones(Bool, n, n)
        graph = GNNGraph(g, graph_type=:dense)
        rof = RandomOrientationFeatures(dim, 0.1f0)
        @test rof(rigid, graph) == reshape(rof(rigid; dims=1), dim, :)
    end

    @testset "Random edges" begin
        dim = 10
        n = 5
        rigid = rand_rigid(Float32, (n,))
        g = rand(Bool, n, n)
        graph = GNNGraph(g, graph_type=:dense)
        rof = RandomOrientationFeatures(dim, 0.1f0)
        @test size(rof(rigid, rigid, graph), 2) == count(g)
        @test rof(rigid, graph) == reshape(rof(rigid; dims=1), dim, :)[:,findall(vec(g))]
        @test rof(rigid, graph) == rof(rigid, rigid, graph)
    end

end
