############################################################################################
"Default Configuration for Metropolis sampler"
struct ConfigMetropolis <: AbstractConfiguration
    "Target Acceptance Rate"
    δ::Float64
    "Default size for tuning iterations in each cycle."
    window::ConfigTuningWindow
    function ConfigMetropolis(
        δ::Float64,
        window::ConfigTuningWindow
    )
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        return new(
            δ,
            window,
        )
    end
end

"""
$(SIGNATURES)
Initialize Metropolis custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{Metropolis},
    objective::Objective,
    proposalconfig::ConfigProposal;
    ## Target Acceptance Rate
    δ=0.234,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window= ConfigTuningWindow(
        [1, 5, 1],
        Int.(floor.([75, 50, 25] * (0.80 / 0.234))),
        [Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
        )
)
    return ConfigMetropolis(
        δ,
        window,
    )
end

############################################################################################
# Export
export ConfigMetropolis
