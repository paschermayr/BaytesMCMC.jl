############################################################################################
"Kinetic Energy in HMC setting."
abstract type KineticEnergy end
"Gaussian Kinetic Energy, independent of position parameter."
abstract type EuclideanKineticEnergy <: KineticEnergy end

#!NOTE: May be converted to immutable struct. But for Riemmann Energy a matrix buffer would be nice to have.
"""
$(TYPEDEF)
Gaussian kinetic energy, which is independent of `q`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct GaussianKineticEnergy{T<:AbstractMatrix,S<:AbstractMatrix} <: EuclideanKineticEnergy
    "Inverse Mass Matrix Σ ~ Posterior Covariance Matrix"
    Σ::T
    "Cholesky decomposition of Mass matrix M, s.t. Mᶜʰᵒˡ*Mᶜʰᵒˡ'=M. Used to generate random draws"
    Mᶜʰᵒˡ::S
    function GaussianKineticEnergy(Σ::T, Σᶜʰᵒˡ::S) where {T,S}
        @argcheck LinearAlgebra.checksquare(Σ) == LinearAlgebra.checksquare(Σᶜʰᵒˡ) #check dimensions
        return new{T,S}(Σ, Σᶜʰᵒˡ)
    end
end

"Gaussian kinetic energy with the given inverse covariance matrix `Σ`."
function GaussianKineticEnergy(Σ::AbstractMatrix)
    return GaussianKineticEnergy(Σ, LinearAlgebra.cholesky(LinearAlgebra.inv(Σ)).L)
end
function GaussianKineticEnergy(Σ::Diagonal)
    return GaussianKineticEnergy(Σ, Diagonal(.√LinearAlgebra.inv.(LinearAlgebra.diag(Σ))))
end
function GaussianKineticEnergy(proposal::Proposal)
    return GaussianKineticEnergy(proposal.Σ, proposal.Σ⁻¹ᶜʰᵒˡ)
end

function update!(energy::GaussianKineticEnergy, Σ::AbstractMatrix, Mᶜʰᵒˡ::AbstractMatrix)
    @pack! energy = Σ, Mᶜʰᵒˡ
    return nothing
end
function update!(energy::GaussianKineticEnergy, proposal::Proposal)
    return update!(energy, proposal.Σ, proposal.Σ⁻¹ᶜʰᵒˡ)
end

#!NOTE: CI Docs break if functor is assigned for documentation.
#"Evaluate kinetic energy with given metric."
function (energy::GaussianKineticEnergy)(ρ::AbstractVector{T}, θᵤ=nothing) where {T<:Real}
    #!NOTE: see Betancourt (2016), Kρ == -log(ρ ∣ θᵤ)
    return LinearAlgebra.dot(ρ, energy.Σ * ρ) / 2 # + constant
end

"Return `p♯ = M⁻¹⋅p`, used for turn diagnostics."
function calculate_ρ♯(K::GaussianKineticEnergy, ρ, θᵤ=nothing)
    return K.Σ * ρ
end

"Calculate the gradient of the logarithm of kinetic energy in momentum ρ."
function ∇K(K::GaussianKineticEnergy, ρ, θᵤ=nothing)
    return calculate_ρ♯(K, ρ)
end

"Generate a random momentum from a kinetic energy at position ρ."
function rand_ρ(_rng::Random.AbstractRNG, K::GaussianKineticEnergy, θᵤ=nothing)
    return K.Mᶜʰᵒˡ * randn(_rng, eltype(K.Mᶜʰᵒˡ), size(K.Mᶜʰᵒˡ, 1))
end

############################################################################################
"""
$(TYPEDEF)
Hamiltonian struct that holds Kinetic energy and log target density function.

# Fields
$(TYPEDFIELDS)
"""
struct Hamiltonian{E<:KineticEnergy,D<:DiffObjective}
    "The kinetic energy specification."
    K::E
    "Differentiable log target density of model, see BaytesDiff."
    objective::D
    function Hamiltonian(K::E, objective::D) where {E<:KineticEnergy,D<:DiffObjective}
        return new{E,D}(K, objective)
    end
end

############################################################################################
"""
$(TYPEDEF)
A point in phase space, consists of a position BaytesDiff.ℓObjectiveResult and a momentum ρ.

# Fields
$(TYPEDFIELDS)
"""
struct PhasePoint{T<:ℓObjectiveResult,S<:Real}
    "BaytesDiff.ℓObjectiveResult container"
    result::T
    "Momentum"
    ρ::Vector{S}
    function PhasePoint(result::T, ρ::Vector{S}) where {T<:ℓObjectiveResult,S<:Real}
        ArgCheck.@argcheck length(ρ) == length(result.θᵤ)
        return new{T,S}(result, ρ)
    end
end

"""
$(SIGNATURES)
Log density for Hamiltonian `H` at `phasepoint`. If `ℓ(q) == -Inf` (rejected), skips the kinetic energy calculation.

# Examples
```julia
```

"""
function ℓdensity(H::Hamiltonian{<:EuclideanKineticEnergy}, phasepoint::PhasePoint)
    @unpack result, ρ = phasepoint
    isfinite(result.ℓθᵤ) || return oftype(result.ℓθᵤ, -Inf)
    K = H.K(ρ)
    #!NOTE: this is the negative(!) log energy
    return result.ℓθᵤ - (isfinite(K) ? K : oftype(K, Inf))
end

function calculate_ρ♯(H::Hamiltonian{<:EuclideanKineticEnergy}, phasepoint::PhasePoint)
    return calculate_ρ♯(H.K, phasepoint.ρ)
end

############################################################################################
"""
$(SIGNATURES)
Take a leapfrog step of length `ϵ` from `phasepoint` along the Hamiltonian `H`.

# Examples
```julia
```

"""
function leapfrog(
    H::Hamiltonian{<:EuclideanKineticEnergy}, phasepoint::PhasePoint, ϵ::T
) where {T<:Real}
    @unpack result, ρ = phasepoint
#    @argcheck isfinite(result.ℓθᵤ) "Internal error: leapfrog called from non-finite log density"
    BaytesDiff.checkfinite(H.objective.objective, result)
    ρ⁰ = ρ + ϵ / 2 * result.∇ℓθᵤ
    θᵤᵖ = result.θᵤ + ϵ * ∇K(H.K, ρ⁰)
    resultᵖ = BaytesDiff.log_density_and_gradient(H.objective, θᵤᵖ)
    ρᵖ = ρ⁰ + ϵ / 2 * resultᵖ.∇ℓθᵤ
    return PhasePoint(resultᵖ, ρᵖ)
end

############################################################################################
# Export
export KineticEnergy,
    EuclideanKineticEnergy,
    GaussianKineticEnergy,
    calculate_ρ♯,
    ∇K,
    rand_ρ,
    Hamiltonian,
    PhasePoint,
    ℓdensity,
    leapfrog
