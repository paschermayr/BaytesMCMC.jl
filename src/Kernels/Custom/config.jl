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
A callable without arguments to initiate a proposal distribution for a Custom sampler.
Can be a closure that captures, i.e., Σ., or a a callable struct:

# Examples
```julia

#1 Method via function
function customconstructor()
    function customdefaultkernel(θ::AbstractVector{T}, ϵ::S) where {S<:Real, T<:Real}
        Nparams = length(θ)
        return MvNormal(θ, T(ϵ) * T(1/Nparams^2) * LinearAlgebra.I(Nparams))
    end
    return customdefaultkernel
end

#2 Method fia callable object in case additional parameter have to be provided. Make sure this is threadsafe in case multiple chains are sampled.
struct _Custom{A,B}
    a::A
    b::B
end
_custom = _Custom(randn(10), I(10))
function (custom::_Custom)()
    @unpack a,b = custom
    μ = deepcopy(a)
    Σ = deepcopy(b)
    dist = MvNormal(μ, Σ)
    function customdefaultkernel(θ::AbstractVector{T}, ϵ::S) where {S<:Real, T<:Real}
        return dist
    end
    return customdefaultkernel
end
```

"""
function customconstructor()
    function customdefaultkernel(θ::AbstractVector{T}, ϵ::S) where {S<:Real, T<:Real}
        Nparams = length(θ)
        return MvNormal(θ, T(ϵ) * T(1/Nparams^2) * LinearAlgebra.I(Nparams))
    end
    return customdefaultkernel
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
    proposal = customconstructor
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
