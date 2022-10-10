############################################################################################
"""
$(TYPEDEF)
HMC - Container used throughout sampling process.

# Fields
$(TYPEDFIELDS)
"""
mutable struct HMC{
    R<:ℓObjectiveResult,D<:AbstractDifferentiableTune,C<:KineticEnergy,T<:BaytesCore.UpdateBool
} <: MCMCKernel
    "Target result stored for caching."
    result::R
    "Differentiation tuning struct."
    diff::D
    "Energy used for Hamiltonian."
    energy::C
    "Tuning struct for discretization steps"
    stepnumber::StepNumberTune{T}
    function HMC(
        result::R, diff::D, energy::C, stepnumber::StepNumberTune{T}
    ) where {
        R<:ℓObjectiveResult,
        D<:AbstractDifferentiableTune,
        C<:KineticEnergy,
        T<:BaytesCore.UpdateBool,
    }
        return new{R,D,C,T}(result, diff, energy, stepnumber)
    end
end
function HMC(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMC(HMC, sym; kwargs...)
end

function update!(kernel::HMC, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    kernel.result = BaytesDiff.log_density_and_gradient(objective, kernel.diff)
    BaytesDiff.checkfinite(objective, kernel.result)
    return nothing
end
function update!(kernel::HMC, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
#Trajectory
"""
$(TYPEDEF)
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that also checks for divergence.

# Fields
$(TYPEDFIELDS)
"""
struct TrajectoryHMC{S<:Hamiltonian,F<:AbstractFloat}
    "Hamiltonian."
    H::S
    "Log density of negative log energy at initial point."
    ℓH₀::F
    "Stepsize for leapfrog."
    ϵ::F
    function TrajectoryHMC(H::S, ℓH₀::F, ϵ::F) where {S<:Hamiltonian,F<:AbstractFloat}
        return new{S,F}(H, ℓH₀, ϵ)
    end
end

function move(trajectory::T, phasepoint::PhasePoint) where {T<:TrajectoryHMC}
    @unpack H, ϵ = trajectory
    return leapfrog(H, phasepoint, ϵ)
end

function checkfinite(trajectory::TrajectoryHMC, phasepoint::PhasePoint)
    return checkfinite(
        trajectory.ℓH₀, ℓdensity(trajectory.H, phasepoint), phasepoint.result
    )
end

############################################################################################
"""
$(SIGNATURES)
Propagate forward one HMC step.

# Examples
```julia
```

"""
function propagate(
    _rng::Random.AbstractRNG, kernel::HMC, tune::MCMCTune, objective::Objective
)
    @unpack ϵ = tune.stepsize
    ## Set new DiffObjective
    diff = DiffObjective(objective, kernel.diff)
    ## If needed, update kinetic energy and proposal leapfrog steps
    update!(kernel.energy, tune.proposal)
    update!(kernel.stepnumber, ϵ)
    ## Fill new Hamiltonian with current Energy and log-posterior adjusted for (latent/new) data
    H = Hamiltonian(kernel.energy, diff)
    ## Create new trajectory and phasepoint, and evaluate Hamiltonian at phasepoint
    phasepoint = PhasePoint(kernel.result, rand_ρ(_rng, H.K))
    ℓH₀ = ℓdensity(H, phasepoint)
    trajectory = TrajectoryHMC(H, ℓH₀, ϵ)
    ## leapfrog and check for divergence
    divergent = false
    for _ in Base.OneTo(kernel.stepnumber.steps)
        phasepoint = move(trajectory, phasepoint)
        if !checkfinite(trajectory, phasepoint)
            divergent = true
            break
        end
    end
    ## Pack container and return output
    accept_statistic = BaytesCore.AcceptStatistic(_rng, ℓdensity(H, phasepoint) - ℓH₀)
    return phasepoint.result,
    divergent, accept_statistic,
    DiagnosticsHMC(tune.stepsize.ϵ, kernel.stepnumber.steps)
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
    _rng::Random.AbstractRNG, kernel::HMC, objective::Objective, Σ::M
) where {M<:AbstractMatrix}
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
#export
export HMC
