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
    "Default size for tuning iterations in each cycle."
    window::ConfigTuningWindow
    "Kinetic Energy used in Hamiltonian: GaussianKineticEnergy"
    energy::K
    function ConfigNUTS(
        δ::Float64,
        window::ConfigTuningWindow,
        energy::K,
    ) where {K<:KineticEnergy}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        return new{K}(
            δ,
            window,
            energy
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
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window= ConfigTuningWindow(
        [1, 5, 1],
        [75, 50, 25],
        [Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()]
        ),
    ## Kinetic energy in Hamiltonian
    energy=GaussianKineticEnergy(
        init(
            objective.model.info.reconstruct.default.output, proposalconfig.metric, length(objective.tagged), proposalconfig.covariance
        )...,
    )
)
    return ConfigNUTS(
        δ,
        window,
        energy
    )
end

############################################################################################
#export
export ConfigNUTS
