module GraphNeuralNetworksExt

using RandomFeatureMaps
using BatchedTransformations
using GraphNeuralNetworks

using RandomFeatureMaps: norms

subt(xi, xj, e) = xj .- xi
function (rof::RandomOrientationFeatures)(T1::Rigid, T2::Rigid, graph::GNNGraph)
    @assert length(batchsize(linear(T1))) == 1 && batchsize(linear(T1)) == batchsize(linear(T2))
    diffs = apply_edges(subt, graph, xj=T1(rof.FA), xi=T2(rof.FB))
    norms(diffs; dims=1)
end

(rof::RandomOrientationFeatures)(T, graph::GNNGraph) = rof(T, T, graph)

end