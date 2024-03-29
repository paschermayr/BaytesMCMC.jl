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

function update!(kernel::Custom, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.result = BaytesDiff.ℓDensityResult(objective)
    BaytesDiff.checkfinite(objective, kernel.result)
    return nothing
end
function update!(kernel::Custom, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
"""
$(TYPEDEF)
Representation of a trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct TrajectoryCustom{P,F<:BaytesDiff.ℓObjectiveResult, T<:Real}
    "Proposal distribution"
    proposal::P
    "Log Target result at initial point, see BaytesDiff."
    result₀::F
    "Discretization size"
    ϵ::T
    function TrajectoryCustom(
        proposal::P, result₀::F, ϵ::T
    ) where {P,F<:BaytesDiff.ℓObjectiveResult, T<:Real}
        return new{P,F,T}(proposal, result₀, ϵ)
    end
end

function move(_rng::Random.AbstractRNG, trajectory::T) where {T<:TrajectoryCustom}
    @unpack result₀, proposal, ϵ = trajectory
    return rand(_rng, proposal(result₀.θᵤ, ϵ))
end

function checkfinite(trajectory::TrajectoryCustom, result::BaytesDiff.ℓObjectiveResult)
    return checkfinite(trajectory.result₀.ℓθᵤ, result.ℓθᵤ, result)
end

############################################################################################
"propagate Custom sampler forward"
function propagate(
    _rng::Random.AbstractRNG, kernel::Custom, tune::MCMCTune, objective::Objective
)
    @unpack result = kernel
    @unpack ϵ = tune.stepsize
    ## Create new trajectory
    trajectory = TrajectoryCustom(kernel.proposal, kernel.result, ϵ)
    ## Make Proposal step
    resultᵖ = BaytesDiff.log_density(objective, move(_rng, trajectory))
    ## Check if proposal diverges
    divergent = !checkfinite(result, resultᵖ)
    if divergent
        return resultᵖ,
        divergent, BaytesCore.AcceptStatistic(zero(eltype(resultᵖ.θᵤ)), false),
        DiagnosticsCustom(ϵ)
    end
    ## Calculate Proposal density and acceptance rate ~ not needed for symmetric proposals
    ℓqᵤ = logpdf(kernel.proposal(resultᵖ.θᵤ, ϵ), result.θᵤ)
    ℓqᵤᵖ = logpdf(kernel.proposal(result.θᵤ, ϵ), resultᵖ.θᵤ)
    ## Pack and return output
    accept_statistic = BaytesCore.AcceptStatistic(_rng, (resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ))
    return resultᵖ, divergent, accept_statistic, DiagnosticsCustom(ϵ)
end

function get_acceptrate(
    _rng::Random.AbstractRNG, kernel::Custom, objective::Objective, Σ::M
) where {M<:AbstractMatrix}
    ## Current logposterior and proposal density
    @unpack result = kernel
    ## Function of stepsize
    return function acceptrate(ϵ::T) where {T<:Real}
        trajectory = TrajectoryCustom(kernel.proposal, kernel.result, ϵ)
        ## Make Proposal step
        resultᵖ = BaytesDiff.log_density(objective, move(_rng, trajectory))
        ℓqᵤ = logpdf(kernel.proposal(resultᵖ.θᵤ, ϵ), result.θᵤ)
        ℓqᵤᵖ = logpdf(kernel.proposal(result.θᵤ, ϵ), resultᵖ.θᵤ)
        ## Return acceptance rate
        return exp((resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ)) #Unbounded accpetance rate
    end
end

############################################################################################
# Export
export Custom
