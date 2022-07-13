############################################################################################
"Default Configuration for Custom sampler"
struct ConfigCustom{F} <: AbstractConfiguration
    "Target Acceptance Rate - ignored for custom sampler."
    δ::Float64
    "Default size for tuning iterations in each cycle - ignored for custom sampler."
    window::ConfigTuningWindow
    "A function/functor of a parameter vector θ that returns a valid Distribution to sample and evaluate from"
    proposal::F
    function ConfigCustom(
        δ::Float64,
        window::ConfigTuningWindow,
        proposal::F
    ) where {F}
        @argcheck 0.0 < δ <= 1.0 "Acceptance rate not bounded between 0 and 1"
        return new{F}(
            δ,
            window,
            proposal
        )
    end
end

############################################################################################
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
function (mixture::MyMixture)(θ, ϵ)
    @unpack weights, Σ1, Σ2 = mixture
    return MixtureModel([MvNormal(θ, ϵ .* Σ1), MvNormal(θ, ϵ .* Σ2)], weights)
end

#Independence sampler via callable struct without extra allocation and no dependency on stepsize
struct MyIndependenceSampler{W}
    dist::W
end
function (sampler::MyIndependenceSampler)(θ, ϵ)
    return sampler.dist
end
```

"""
function customdefaultkernel(θ::AbstractVector{T}, ϵ::S) where {S<:Real, T<:Real}
    Nparams = length(θ)
    return MvNormal(θ, ϵ .* T(1/Nparams^2) .* LinearAlgebra.I(Nparams))
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
    objective::Objective,
    proposalconfig::ConfigProposal;
    ## Target Acceptance Rate
    δ=0.234,
    ## MCMC Phase tune variables based on standard HMC best practice target acceptance rate of 0.80
    window= ConfigTuningWindow(
        [1, 5, 1],
        Int.(floor.([75, 50, 25] * (0.80 / 0.234))),
        [Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
        ),
    ## Custom distribution
    proposal = customdefaultkernel
)
    return ConfigCustom(
        δ,
        window,
        proposal
    )
end

############################################################################################
# Export
export ConfigCustom, customdefaultkernel
