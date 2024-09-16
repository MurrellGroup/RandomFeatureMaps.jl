module GraphNeuralNetworksExt

using RandomFeatureMaps
using BatchedTransformations
using GraphNeuralNetworks

subt(xi, xj, e) = xj .- xi
function (rof::RandomOrientationFeatures)(rigid::Rigid, graph::GNNGraph)
    points1 = rigid * rof.FA
    points2 = rigid * rof.FB
    diffs = apply_edges(subt, graph, xi=points2, xj=points1)
    return dropdims(sqrt.(sum(abs2, diffs; dims=1)); dims=1)
end

# deprecated
(rof::RandomOrientationFeatures)(graph::GNNGraph, rigid) = rof(rigid, graph)

end