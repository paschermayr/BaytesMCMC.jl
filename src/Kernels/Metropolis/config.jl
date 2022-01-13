############################################################################################
"Default Configuration for Metropolis sampler"
struct ConfigMetropolis{M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool} <:
       AbstractConfiguration
    ## Global targets
    "Target Acceptance Rate"
    δ::Float64
    ## Stepsize adaption
    "Step size adaption"
    stepsizeadaption::A
    "Discretization size"
    ϵ::Float64
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

    function ConfigMetropolis(
        δ::Float64,
        stepsizeadaption::A,
        ϵ::Float64,
        proposaladaption::B,
        metric::M,
        shrinkage::Float64,
        window::Vector{Int64},
        buffer::Vector{Int64},
        phasenames::Vector{SamplingPhase},
    ) where {M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        @argcheck ϵ > 0.0 "Discretization size has to be positive"
        @argcheck 0.0 <= shrinkage <= 1.0 "Shrinkage not bounded between 0 and 1"
        @argcheck size(window, 1) == size(buffer, 1) "Window and Buffer size different"
        @argcheck (size(window, 1) + 1) == size(phasenames, 1) "Window and phasename (without Exploration) size different"
        return new{M,A,B}(
            δ,
            stepsizeadaption,
            ϵ,
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
Initialize Metropolis custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{Metropolis},
    objective::Objective;
    ## Target Acceptance Rate
    δ=0.234,
    ## Discretization size variables
    stepsizeadaption=BaytesCore.UpdateTrue(),
    ϵ=0.05,
    ## Posterior Covariance adaption variables
    proposaladaption=BaytesCore.UpdateTrue(),
    metric=MDiagonal(),
    shrinkage=0.05,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window=[1, 5, 1],
    buffer=Int.(floor.([75, 50, 25] * (0.80 / 0.234))),
    phasenames=[Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
)
    return ConfigMetropolis(
        δ,
        stepsizeadaption,
        ϵ,
        proposaladaption,
        metric,
        shrinkage,
        window,
        buffer,
        phasenames,
    )
end

############################################################################################
# Export
export ConfigMetropolis
