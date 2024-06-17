using RandomFeatureMaps
using Documenter

DocMeta.setdocmeta!(RandomFeatureMaps, :DocTestSetup, :(using RandomFeatureMaps); recursive=true)

makedocs(;
    modules=[RandomFeatureMaps],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="RandomFeatureMaps.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/RandomFeatureMaps.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/RandomFeatureMaps.jl",
    devbranch="main",
)
