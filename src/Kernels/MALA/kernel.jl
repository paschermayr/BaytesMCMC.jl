############################################################################################
"""
$(TYPEDEF)
MALA algorithm container.

# Fields
$(TYPEDFIELDS)
"""
mutable struct MALA{M<:ℓGradientResult,D<:AbstractDifferentiableTune} <: MCMCKernel
    "Cached Result of last propagation step."
    result::M
    "Differentiation tuning container"
    diff::D
    function MALA(
        result::M, diff::D
    ) where {M<:ℓObjectiveResult,D<:AbstractDifferentiableTune}
        return new{M,D}(result, diff)
    end
end
function MALA(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMC(MALA, sym; kwargs...)
end

function update!(kernel::MALA, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    kernel.result = BaytesDiff.log_density_and_gradient(objective, kernel.diff)
    BaytesDiff.checkfinite(objective, kernel.result)
    return nothing
end
function update!(kernel::MALA, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end
############################################################################################
#Trajectory
"""
$(TYPEDEF)
Representation of a trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct TrajectoryMALA{S<:AbstractMatrix,F<:BaytesDiff.ℓObjectiveResult,T<:AbstractFloat}
    "Proposal Covariance"
    Σ::S
    "Log Target result at initial point, see BaytesDiff."
    result₀::F
    "Discretization size"
    ϵ::T
    function TrajectoryMALA(
        Σ::S, result₀::F, ϵ::T
    ) where {S<:AbstractMatrix,F<:BaytesDiff.ℓObjectiveResult,T<:AbstractFloat}
        return new{S,F,T}(Σ, result₀, ϵ)
    end
end

function move(_rng::Random.AbstractRNG, trajectory::T) where {T<:TrajectoryMALA}
    @unpack result₀, Σ, ϵ = trajectory
    return result₀.θᵤ +
           (ϵ / 2 * Σ * result₀.∇ℓθᵤ) +
           sqrt(ϵ) *
           LinearAlgebra.cholesky(Σ).L *
           randn(_rng, eltype(result₀.θᵤ), size(result₀.θᵤ, 1))
end

function checkfinite(trajectory::TrajectoryMALA, result::BaytesDiff.ℓObjectiveResult)
    return checkfinite(trajectory.result₀.ℓθᵤ, result.ℓθᵤ, result)
end

############################################################################################
"""
$(SIGNATURES)
Propagate forward one MALA step.

# Examples
```julia
```

"""
function propagate(
    _rng::Random.AbstractRNG, kernel::MALA, tune::MCMCTune, objective::Objective
)
    @unpack result = kernel
    @unpack Σ = tune.proposal
    @unpack ϵ = tune.stepsize
    ## Create new trajectory
    trajectory = TrajectoryMALA(Σ, kernel.result, ϵ)
    ## Make Proposal step
    resultᵖ = BaytesDiff.log_density_and_gradient(
        objective, kernel.diff, move(_rng, trajectory)
    )
    ## Check if proposal diverges
    divergent = !checkfinite(trajectory, resultᵖ) #!checkfinite(result, resultᵖ)
    if divergent
        return resultᵖ,
        divergent, BaytesCore.AcceptStatistic(zero(typeof(ϵ)), false),
        DiagnosticsMALA(ϵ)
    end
    ## Calculate proposal density and acceptance rate
    ℓqₜ = logpdf(MvNormal(resultᵖ.θᵤ + ϵ / 2 * Σ * resultᵖ.∇ℓθᵤ, ϵ * Σ), result.θᵤ)
    ℓqₜᵖ = logpdf(MvNormal(result.θᵤ + ϵ / 2 * Σ * result.∇ℓθᵤ, ϵ * Σ), resultᵖ.θᵤ)
    accept_statistic = BaytesCore.AcceptStatistic(_rng, (resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqₜ - ℓqₜᵖ))
    ## Pack and return output
    return resultᵖ, divergent, accept_statistic, DiagnosticsMALA(ϵ)
end

############################################################################################
function get_acceptrate(
    _rng::Random.AbstractRNG, kernel::MALA, objective::Objective, Σ::M
) where {M<:AbstractMatrix} # = LinearAlgebra.Diagonal(ones(length(objective.tagged)))
    ## Calculate current logposterior and proposal density
    @unpack result = kernel
    ## Function of stepsize
    return function acceptrate(ϵ::T) where {T<:Real}
        trajectory = TrajectoryMALA(Σ, result, ϵ)
        ## Make Proposal step
        resultᵖ = BaytesDiff.log_density_and_gradient(
            objective, kernel.diff, move(_rng, trajectory)
        )
        ## Calculate proposal density and acceptance rate
        ℓqₜ = logpdf(MvNormal(resultᵖ.θᵤ + ϵ / 2 * Σ * resultᵖ.∇ℓθᵤ, ϵ * Σ), result.θᵤ)
        ℓqₜᵖ = logpdf(MvNormal(result.θᵤ + ϵ / 2 * Σ * result.∇ℓθᵤ, ϵ * Σ), resultᵖ.θᵤ)
        ## Return acceptance rate
        return exp((resultᵖ.ℓθᵤ - result.ℓθᵤ) + (ℓqₜ - ℓqₜᵖ)) #Unbounded accpetance rate
    end
end

############################################################################################
# Export
export MALA, ConfigMALA
