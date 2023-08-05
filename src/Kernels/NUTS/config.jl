############################################################################################
"""
$(TYPEDEF)
Default Configuration for NUTS sampler.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigNUTS{K<:KineticEnergy} <: AbstractConfiguration
    ## Global targets
    "Target Acceptance Rate"
    δ::Float64
    "Maximum tree depth."
    max_depth::Int64
    "Default size for tuning iterations in each cycle."
    window::ConfigTuningWindow
    "Kinetic Energy used in Hamiltonian: GaussianKineticEnergy"
    energy::K
    "Differentiable order for objective function needed to run proposal step"
    difforder::BaytesDiff.DiffOrderOne
    function ConfigNUTS(
        δ::Float64,
        max_depth::Int64,
        window::ConfigTuningWindow,
        energy::K,
        difforder::BaytesDiff.DiffOrderOne
    ) where {K<:KineticEnergy}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        @argcheck 0 < max_depth <= MAX_DIRECTIONS_DEPTH "max tree depth bounded by 32, set a value lower than this."
        return new{K}(
            δ,
            max_depth,
            window,
            energy,
            difforder
        )
    end
end
"""
$(SIGNATURES)
Initialize NUTS custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{NUTS},
    objective::Objective,
    proposalconfig::ConfigProposal;
    ## Target Acceptance Rate
    δ=0.80,
    max_depth = 10,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window= ConfigTuningWindow(
        [1, 5, 1],
        [75, 50, 25],
        [Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()]
        ),
    ## Kinetic energy in Hamiltonian
    energy=GaussianKineticEnergy(
        init(
            objective.model.info.reconstruct.default.output, proposalconfig.metric, ModelWrappers.length_unconstrained(objective.tagged), proposalconfig.covariance
        )...,
    )
)
    return ConfigNUTS(
        δ,
        max_depth,
        window,
        energy,
        BaytesDiff.DiffOrderOne()
    )
end

############################################################################################
#export
export ConfigNUTS
