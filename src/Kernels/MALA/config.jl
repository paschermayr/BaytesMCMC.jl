############################################################################################
"""
$(TYPEDEF)
Default Configuration for MALA sampler.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigMALA <: AbstractConfiguration
    "Target Acceptance Rate"
    δ::Float64
    "Default size for tuning iterations in each cycle."
    window::ConfigTuningWindow
    "Differentiable order for objective function needed to run proposal step"
    difforder::BaytesDiff.DiffOrderOne
    function ConfigMALA(
        δ::Float64,
        window::ConfigTuningWindow,
        difforder::BaytesDiff.DiffOrderOne
    )
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        return new(
            δ,
            window,
            difforder
        )
    end
end

"""
$(SIGNATURES)
Initialize Mala custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{MALA},
    objective::Objective,
    proposalconfig::ConfigProposal;
    ## Target Acceptance Rate
    δ=0.574,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window= ConfigTuningWindow(
        [1, 5, 1],
        Int.(floor.([75, 50, 25] * (0.80 / 0.574))),
        [Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
        )
)
    return ConfigMALA(
        δ,
        window,
        BaytesDiff.DiffOrderOne()
    )
end

############################################################################################
#export
export ConfigMALA
