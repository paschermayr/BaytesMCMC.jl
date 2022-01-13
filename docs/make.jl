using BaytesMCMC
using Documenter

DocMeta.setdocmeta!(BaytesMCMC, :DocTestSetup, :(using BaytesMCMC); recursive=true)

makedocs(;
    modules=[BaytesMCMC],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesMCMC.jl/blob/{commit}{path}#{line}",
    sitename="BaytesMCMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesMCMC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesMCMC.jl",
    devbranch="main",
)
