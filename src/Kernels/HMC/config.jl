############################################################################################
"""
$(TYPEDEF)

Default configuration for stepsize adaption.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigStepnumber{A<:BaytesCore.UpdateBool}
    "Step Number adaption"
    stepnumberadaption::A
    "Initial number of Steps"
    steps::Int64
    "Maximal number of steps"
    maxsteps::Int64
    "Desired Integration time"
    ∫dt::Float64
    function ConfigStepnumber(;
        stepnumberadaption=BaytesCore.UpdateTrue(),
        steps=10,
        maxsteps=100,
        ∫dt=1.0,
        )
        @argcheck steps <= maxsteps "Upper bound for L is maxsteps."
        @argcheck 0.0 < ∫dt "Integration time needs to be positive."
        return new{typeof(stepnumberadaption)}(
            stepnumberadaption, steps, maxsteps,∫dt
            )
    end
end

############################################################################################
"""
$(TYPEDEF)
Default Configuration for HMC sampler.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigHMC{
    K<:KineticEnergy,S<:ConfigStepnumber
} <: AbstractConfiguration
    "Target Acceptance Rate"
    δ::Float64
    "Default size for tuning iterations in each cycle."
    window::ConfigTuningWindow
    "Kinetic Energy used in Hamiltonian: GaussianKineticEnergy"
    energy::K
    "Step Number adaption"
    stepnumber::S
    "Differentiable order for objective function needed to run proposal step"
    difforder::BaytesDiff.DiffOrderOne
    function ConfigHMC(
        δ::Float64,
        window::ConfigTuningWindow,
        energy::K,
        stepnumber::S,
        difforder::BaytesDiff.DiffOrderOne
    ) where {K<:KineticEnergy,S<:ConfigStepnumber}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1."
        return new{K,S}(
            δ,
            window,
            energy,
            stepnumber,
            BaytesDiff.DiffOrderOne()
        )
    end
end

"""
$(SIGNATURES)
Initialize HMC custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{HMC},
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
        init(objective.model.info.reconstruct.default.output, proposalconfig.metric, ModelWrappers.length_unconstrained(objective.tagged), proposalconfig.covariance)...,
    ),
    stepnumber = ConfigStepnumber()
)
    return ConfigHMC(
        δ,
        window,
        energy,
        stepnumber,
        BaytesDiff.DiffOrderOne()
    )
end

############################################################################################
#export
export ConfigHMC, ConfigStepnumber
