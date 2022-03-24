############################################################################################
"""
$(TYPEDEF)
Custom algorithm container.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Custom{R<:ℓDensityResult, P} <: MCMCKernel
    "Cached Result of last propagation step."
    result::R
    "Callable struct/closure/function of `result.θᵤ` that returns proposal distribution."
    proposal::P
    function Custom(result::R, proposal::P) where {R<:ℓObjectiveResult, P}
        return new{R, P}(result, proposal)
    end
end
function Custom(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMC(Custom, sym; kwargs...)
end

function update!(kernel::Custom, objective::Objective, up::U) where {U<:UpdateBool}
    return nothing
end

############################################################################################
"""
$(TYPEDEF)
Representation of a trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct TrajectoryCustom{P,F<:ModelWrappers.ℓObjectiveResult}
    "Proposal distribution"
    proposal::P
    "Log Target result at initial point, see ModelWrappersesTools."
    result₀::F
    function TrajectoryCustom(
        proposal::P, result₀::F
    ) where {P,F<:ModelWrappers.ℓObjectiveResult}
        return new{P,F}(proposal, result₀)
    end
end

function move(_rng::Random.AbstractRNG, trajectory::T) where {T<:TrajectoryCustom}
    @unpack result₀, proposal = trajectory
    return rand(_rng, proposal(result₀.θᵤ))
end

function checkfinite(trajectory::TrajectoryCustom, result::ModelWrappers.ℓObjectiveResult)
    return checkfinite(trajectory.result₀.ℓθᵤ, result.ℓθᵤ, result)
end

############################################################################################
"propagate Custom sampler forward"
function propagate(
    _rng::Random.AbstractRNG, kernel::Custom, tune::MCMCTune, objective::Objective
)
    @unpack result = kernel
    ## Create new trajectory
    trajectory = TrajectoryCustom(kernel.proposal, kernel.result)
    ## Make Proposal step
    resultᵖ = ModelWrappers.log_density(objective, move(_rng, trajectory))
    ## Check if proposal diverges
    divergent = !checkfinite(result, resultᵖ)
    if divergent
        return resultᵖ,
        divergent, BaytesCore.AcceptStatistic(zero(eltype(resultᵖ.θᵤ)), false),
        DiagnosticsCustom()
    end
    ## Calculate Proposal density and acceptance rate ~ not needed for symmetric proposals
    ℓqᵤ = logpdf(kernel.proposal(resultᵖ.θᵤ), result.θᵤ)
    ℓqᵤᵖ = logpdf(kernel.proposal(result.θᵤ), resultᵖ.θᵤ)
    ## Pack and return output
    accept_statistic = BaytesCore.AcceptStatistic(_rng, (resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ))
    return resultᵖ, divergent, accept_statistic, DiagnosticsCustom()
end

function get_acceptrate(
    _rng::Random.AbstractRNG, kernel::Custom, objective::Objective, Σ::M
) where {M<:AbstractMatrix}
    ## Current logposterior and proposal density
    @unpack result = kernel
    ## Function of stepsize
    return function acceptrate(ϵ::T) where {T<:Real}
        trajectory = TrajectoryCustom(kernel.proposal, kernel.result)
        ## Make Proposal step
        resultᵖ = ModelWrappers.log_density(objective, move(_rng, trajectory))
        ℓqᵤ = logpdf(kernel.proposal(resultᵖ.θᵤ), result.θᵤ)
        ℓqᵤᵖ = logpdf(kernel.proposal(result.θᵤ), resultᵖ.θᵤ)
        ## Return acceptance rate
        return exp((resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ)) #Unbounded accpetance rate
    end
end

############################################################################################
# Export
export Custom