############################################################################################
"""
$(TYPEDEF)
Default Configuration for HMC sampler.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigHMC{
    K<:KineticEnergy,M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool,C<:BaytesCore.UpdateBool
} <: AbstractConfiguration
    ## Global targets
    "Target Acceptance Rate"
    δ::Float64
    ## Stepsize adaption
    "Step size adaption"
    stepsizeadaption::A
    "Discretization size"
    ϵ::Float64
    ## Kinetic Energy
    "Kinetic Energy used in Hamiltonian: GaussianKineticEnergy"
    energy::K
    ## Number of Leapfrog steps
    "Step Number adaption"
    stepnumberadaption::C
    "Initial number of Steps"
    steps::Int64
    "Maximal number of steps"
    maxsteps::Int64
    "Desired Integration time"
    ∫dt::Float64
    ## Posterior covariance adaption
    "Posterior Covariance adaption"
    proposaladaption::B
    "Covariance estimate metric: MDense(), MDiagonal(), MUnit()"
    metric::M
    "Shrinkage parameter towards Diagonal Matrix with equal variance"
    shrinkage::Float64
    ## MCMC Tuning windows
    "MCMC Phase tuning window lengths, i.e: 5 = 5 repeats of second window in phasenames"
    window::Vector{Int64}
    "(Increasing) Length of windows, i.e.: if window=3, buffer=10 -> total window length: 10-20-30"
    buffer::Vector{Int64}
    "Name of phasenames, currently supported: Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()"
    phasenames::Vector{SamplingPhase}

    function ConfigHMC(
        δ::Float64,
        stepsizeadaption::A,
        ϵ::Float64,
        energy::K,
        stepnumberadaption::C,
        steps::Int64,
        maxsteps::Int64,
        ∫dt::Float64,
        proposaladaption::B,
        metric::M,
        shrinkage::Float64,
        window::Vector{Int64},
        buffer::Vector{Int64},
        phasenames::Vector{SamplingPhase},
    ) where {K,M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool,C<:BaytesCore.UpdateBool}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1."
        @argcheck ϵ > 0.0 "Discretization size has to be positive."
        @argcheck steps <= maxsteps "Upper bound for L is maxsteps."
        @argcheck 0.0 <= shrinkage <= 1.0 "Shrinkage not bounded between 0 and 1."
        @argcheck 0.0 < ∫dt "Integration time needs to be positive."
        @argcheck size(window, 1) == size(buffer, 1) "Window and Buffer size different."
        @argcheck (size(window, 1) + 1) == size(phasenames, 1) "Window and phasename (without Exploration) size different."
        return new{K,M,A,B,C}(
            δ,
            stepsizeadaption,
            ϵ,
            energy,
            stepnumberadaption,
            steps,
            maxsteps,
            ∫dt,
            proposaladaption,
            metric,
            shrinkage,
            window,
            buffer,
            phasenames,
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
    objective::Objective;
    ## Target Acceptance Rate
    δ=0.80,
    ## Discretization size variables
    stepsizeadaption=BaytesCore.UpdateTrue(),
    ϵ=0.10,
    ## Step number variables
    stepnumberadaption=BaytesCore.UpdateTrue(),
    steps=10,
    maxsteps=100,
    ∫dt=1.0,
    ## Posterior Covariance adaption variables
    proposaladaption=BaytesCore.UpdateTrue(),
    metric=MDiagonal(),
    shrinkage=0.05,
    ## Kinetic energy in Hamiltonian
    energy=GaussianKineticEnergy(
        init(objective.model.info.flattendefault.output, metric, length(objective.tagged))...,
    ),
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window=[1, 5, 1],
    buffer=[75, 50, 25],
    phasenames=[Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
)
    return ConfigHMC(
        δ,
        stepsizeadaption,
        ϵ,
        energy,
        stepnumberadaption,
        steps,
        maxsteps,
        ∫dt,
        proposaladaption,
        metric,
        shrinkage,
        window,
        buffer,
        phasenames,
    )
end

############################################################################################
#export
export ConfigHMC
