############################################################################################
"Default Configuration for Custom sampler"
struct ConfigCustom{F,M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool} <:
       AbstractConfiguration
    "A function of a parameter vector θ that returns a valid Distribution to sample and evaluate from"
    proposal::F
#!NOTE: Most of the configuration settings are ignored for custom sampler
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
    function ConfigCustom(
        proposal::F,
        δ::Float64,
        stepsizeadaption::A,
        ϵ::Float64,
        proposaladaption::B,
        metric::M,
        shrinkage::Float64,
        window::Vector{Int64},
        buffer::Vector{Int64},
        phasenames::Vector{SamplingPhase},
    ) where {F, M<:MatrixMetric,A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        @argcheck ϵ > 0.0 "Discretization size has to be positive"
        @argcheck 0.0 <= shrinkage <= 1.0 "Shrinkage not bounded between 0 and 1"
        @argcheck size(window, 1) == size(buffer, 1) "Window and Buffer size different"
        @argcheck (size(window, 1) + 1) == size(phasenames, 1) "Window and phasename (without Exploration) size different"
        return new{F, M,A,B}(
            proposal,
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
A placeholder function to configure a custom proposal distribution.
Can also be a closure that captures, i.e., Σ., or a a callable struct:

# Examples
```julia

#Mixture via callable struct
struct MyMixture{W, F, T}
    weights::W
    Σ1::F
    Σ2::T
end
function (mixture::MyMixture)(θ)
    @unpack weights, Σ1, Σ2 = mixture
    return MixtureModel([MvNormal(θ, Σ1), MvNormal(θ, Σ2)], weights)
end

#Independence sampler via callable struct without extra allocation
struct MyIndependenceSampler{W}
    dist::W
end
function (sampler::MyIndependenceSampler)(θ)
    return sampler.dist
end
```

"""
function customdefaultkernel(θ::AbstractVector{T}) where {T<:Real}
    Nparams = length(θ)
    return MvNormal(θ, T(1/Nparams^2) * LinearAlgebra.I(Nparams))
end

"""
$(SIGNATURES)
Initialize Custom custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    mcmc::Type{Custom},
    objective::Objective;
    ## Custom distribution
    proposal = customdefaultkernel,
    ## Target Acceptance Rate
    δ=0.234,
    ## Discretization size variables
    stepsizeadaption=BaytesCore.UpdateFalse(),
    ϵ=0.05,
    ## Posterior Covariance adaption variables
    proposaladaption=BaytesCore.UpdateFalse(),
    metric=MDiagonal(),
    shrinkage=0.05,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window=[1, 1, 1],
    buffer=Int.(floor.([75, 50, 25])),
    phasenames=[Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
)
    return ConfigCustom(
        proposal,
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
export ConfigCustom
