############################################################################################
"""
$(TYPEDEF)
NUTS - Container used throughout sampling process.

# Fields
$(TYPEDFIELDS)
"""
mutable struct NUTS{R<:ℓObjectiveResult,D<:AbstractDifferentiableTune,C<:KineticEnergy} <:
               MCMCKernel
    "Target result stored for caching."
    result::R
    "Differentiation tuning struct."
    diff::D
    "Energy used for Hamiltonian."
    energy::C
    function NUTS(
        result::R, diff::D, energy::C
    ) where {R<:ℓObjectiveResult,D<:AbstractDifferentiableTune,C<:KineticEnergy}
        return new{R,D,C}(result, diff, energy)
    end
end
function NUTS(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMC(NUTS, sym; kwargs...)
end

function update!(kernel::NUTS, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    kernel.result = ModelWrappers.log_density_and_gradient(objective, kernel.diff)
    return nothing
end
function update!(kernel::NUTS, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
No-U-turn Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at evaluated
log density position `Q`, using stepsize `ϵ`. Parameters of `kernel` govern the details
of tree construction. Return two values, the new evaluated log density position, and tree statistics.

# Examples
```julia
```

"""
function sample_tree(
    _rng::Random.AbstractRNG,
    trajectory::TrajectoryNUTS,
    phasepoint::PhasePoint;
    directions=rand(_rng, Directions),
)
    phasepointᵖ, v, termination, depth = sample_trajectory(
        _rng, trajectory, phasepoint, directions
    )
    tree_statistics = DiagnosticsNUTS(
        ℓdensity(trajectory.H, phasepointᵖ),
        depth,
        termination,
        acceptance_rate(v),
        trajectory.ϵ,
        v.steps,
        directions,
    )
    return phasepointᵖ, tree_statistics
end

############################################################################################
"""
$(SIGNATURES)
Propagate forward one proposal step.

# Examples
```julia
```

"""
function propagate(
    _rng::Random.AbstractRNG, kernel::NUTS, tune::MCMCTune, objective::Objective
)
    @unpack ϵ = tune.stepsize
    ## Set new DiffObjective
    diff = DiffObjective(objective, kernel.diff)
    ## If needed, update kinetic energy
    update!(kernel.energy, tune.proposal)
    H = Hamiltonian(kernel.energy, diff)
    ## Create new trajectory and phasepoint, and evaluate Hamiltonian at phasepoint
    phasepoint = PhasePoint(kernel.result, rand_ρ(_rng, H.K))
    trajectory = TrajectoryNUTS(H, ℓdensity(H, phasepoint), ϵ)
    ## Sample from NUTS trajectory
    phasepointᵖ, sampler_statistic = sample_tree(_rng, trajectory, phasepoint)
    ## Pack container and return output
    return phasepointᵖ.result,
    is_divergent(sampler_statistic.termination),
    AcceptStatistic(sampler_statistic.acceptance_rate, true),
    sampler_statistic
end

############################################################################################
"""
$(SIGNATURES)
Tune initial step size.

# Examples
```julia
```

"""
function get_acceptrate(
    _rng::Random.AbstractRNG, kernel::NUTS, objective::Objective, Σ::M
) where {M<:AbstractMatrix} # LinearAlgebra.Diagonal(ones(length(objective.tagged)))
    ## Update Hamiltonian with current Σ and log-posterior adjusted for (latent/new) data
    H = Hamiltonian(kernel.energy, DiffObjective(objective, kernel.diff))
    ## Create new trajectory and phasepoint, and evaluate Hamiltonian at phasepoint
    phasepoint = PhasePoint(kernel.result, rand_ρ(_rng, H.K))
    H₀ = ℓdensity(H, phasepoint)
    ## Function of stepsize
    return function acceptrate(ϵ::T) where {T<:Real}
        # Make leapfrog step
        phasepointᵖ = leapfrog(H, phasepoint, ϵ)
        # Calculate acceptance rate
        return exp(ℓdensity(H, phasepointᵖ) - H₀)
    end
end

############################################################################################
# Export
export NUTS, sample_tree
