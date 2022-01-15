############################################################################################
"""
$(TYPEDEF)
Metropolis algorithm container.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Metropolis{R<:ℓDensityResult} <: MCMCKernel
    "Cached Result of last propagation step."
    result::R
    function Metropolis(result::R) where {R<:ℓObjectiveResult}
        return new{R}(result)
    end
end
function Metropolis(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMC(Metropolis, sym; kwargs...)
end

function update!(kernel::Metropolis, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.result = ModelWrappers.ℓDensityResult(objective)
    return nothing
end
function update!(kernel::Metropolis, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end
############################################################################################
"""
$(TYPEDEF)
Representation of a trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct TrajectoryMetropolis{
    S<:AbstractMatrix,F<:ModelWrappers.ℓObjectiveResult,T<:AbstractFloat
}
    "Proposal Covariance"
    Σ::S
    "Log Target result at initial point, see ModelWrappersesTools."
    result₀::F
    "Discretization size"
    ϵ::T
    function TrajectoryMetropolis(
        Σ::S, result₀::F, ϵ::T
    ) where {S<:AbstractMatrix,F<:ModelWrappers.ℓObjectiveResult,T<:AbstractFloat}
        return new{S,F,T}(Σ, result₀, ϵ)
    end
end

function move(_rng::Random.AbstractRNG, trajectory::T) where {T<:TrajectoryMetropolis}
    @unpack result₀, Σ, ϵ = trajectory
    return rand(_rng, MvNormal(result₀.θᵤ, ϵ .* Σ))
end

function checkfinite(
    trajectory::TrajectoryMetropolis, result::ModelWrappers.ℓObjectiveResult
)
    return checkfinite(trajectory.result₀.ℓθᵤ, result.ℓθᵤ, result)
end

############################################################################################
"propagate Metropolis sampler forward"
function propagate(
    _rng::Random.AbstractRNG, kernel::Metropolis, tune::MCMCTune, objective::Objective
)
    @unpack result = kernel
    @unpack Σ = tune.proposal
    @unpack ϵ = tune.stepsize
    ## Create new trajectory
    trajectory = TrajectoryMetropolis(Σ, kernel.result, ϵ)
    ## Make Proposal step
    resultᵖ = ModelWrappers.log_density(objective, move(_rng, trajectory))
    ## Check if proposal diverges
    divergent = !checkfinite(result, resultᵖ)
    if divergent
        return resultᵖ,
        divergent, BaytesCore.AcceptStatistic(zero(typeof(ϵ)), false),
        DiagnosticsMetropolis(ϵ)
    end
    ## Calculate Proposal density and acceptance rate
    ℓqᵤ = logpdf(MvNormal(resultᵖ.θᵤ, ϵ * Σ), result.θᵤ)
    ℓqᵤᵖ = logpdf(MvNormal(result.θᵤ, ϵ * Σ), resultᵖ.θᵤ)
    ## Pack and return output
    accept_statistic = BaytesCore.AcceptStatistic(_rng, (resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ))
    return resultᵖ, divergent, accept_statistic, DiagnosticsMetropolis(ϵ)
end

function get_acceptrate(
    _rng::Random.AbstractRNG, kernel::Metropolis, objective::Objective, Σ::M
) where {M<:AbstractMatrix}
    ## Current logposterior and proposal density
    @unpack result = kernel
    ## Function of stepsize
    return function acceptrate(ϵ::T) where {T<:Real}
        trajectory = TrajectoryMetropolis(Σ, result, ϵ)
        ## Make Proposal step
        resultᵖ = ModelWrappers.log_density(objective, move(_rng, trajectory))
        ℓqᵤ = logpdf(MvNormal(resultᵖ.θᵤ, ϵ * Σ), result.θᵤ)
        ℓqᵤᵖ = logpdf(MvNormal(result.θᵤ, ϵ * Σ), resultᵖ.θᵤ)
        ## Return acceptance rate
        return exp((resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqᵤ - ℓqᵤᵖ)) #Unbounded accpetance rate
    end
end

############################################################################################
# Export
export Metropolis
