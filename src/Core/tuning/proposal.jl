#=
NOTES:
- Proposal will be updated after the end of each cycle in Warmup and Adaptionˢˡᵒʷ.
- Proposal is kept constant in Adaptionᶠᵃˢᵗ, where final stepsize is tuned.
- Proposal is kept constant in Exploration.
=#
############################################################################################
"""
$(TYPEDEF)

Choice for Posterior Covariance Matrix adaption.

# Fields
$(TYPEDFIELDS)
"""
abstract type MatrixMetric end
struct MDense <: MatrixMetric end
struct MDiagonal <: MatrixMetric end
struct MUnit <: MatrixMetric end

############################################################################################
"""
$(TYPEDEF)

Default configuration for posterior Covariance Matrix adaption.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigProposal{A<:BaytesCore.UpdateBool, M<:MatrixMetric}
    "Boolean if Posterior covariance for proposal steps is adapted."
    proposaladaption::A
    "Covariance estimate metric: MDense(), MDiagonal(), MUnit()"
    metric::M
    "Shrinkage parameter towards Diagonal Matrix with equal variance"
    shrinkage::Float64
    function ConfigProposal(;
        proposaladaption = BaytesCore.UpdateTrue(),
        metric = MDiagonal(),
        shrinkage = 0.05
        )
        @argcheck 0.0 <= shrinkage <= 1.0 "Shrinkage not bounded between 0 and 1"
        return new{typeof(proposaladaption), typeof(metric)}(
            proposaladaption, metric, shrinkage
            )
    end
end

############################################################################################
#Wrapper to generate Covariance buffers
function init(type::Type{T}, metric::MDense, param_length::Int64) where {T<:Real}
    Σ = LinearAlgebra.Symmetric(zeros(T, param_length, param_length) + LinearAlgebra.I)
    Σ⁻¹ᶜʰᵒˡ = LinearAlgebra.cholesky(LinearAlgebra.inv(Σ)).L
    return Σ, Σ⁻¹ᶜʰᵒˡ
end
function init(
    type::Type{T}, metric::M, param_length::Int64
) where {T<:Real,M<:Union{MDiagonal,MUnit}}
    Σ = LinearAlgebra.Diagonal(fill(T(1.0), param_length))
    Σ⁻¹ᶜʰᵒˡ = Diagonal(.√LinearAlgebra.inv.(LinearAlgebra.diag(Σ)))
    return Σ, Σ⁻¹ᶜʰᵒˡ
end

############################################################################################
"""
$(TYPEDEF)

Mass and Covariance Matrix specification for MCMC sampler, relevant for Euclidean Metric.

# Fields
$(TYPEDFIELDS)
"""
struct MatrixTune{A<:MatrixMetric}
    "Dense, Diagonal or Unit Mass Matrix."
    metric::A
    "Shrinkage parameter for Covariance estimation."
    shrinkage::Float64
    function MatrixTune(metric::A, shrinkage::Float64) where {A<:MatrixMetric}
        return new{A}(metric, shrinkage)
    end
end

############################################################################################
"""
$(SIGNATURES)
Calculate regularized Covariance Matrix.

# Examples
```julia
```

"""
function get_Σ(Σ::D, shrinkage::T) where {D<:Union{Diagonal,Symmetric},T<:Real}
    return (1 - shrinkage) * Σ + shrinkage * UniformScaling(max(1e-3, median(diag(Σ))))
end

############################################################################################
"""
$(TYPEDEF)

Proposal distribution container.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Proposal{
    A<:BaytesCore.UpdateBool,T<:AbstractFloat,P<:AbstractMatrix,C<:AbstractMatrix,M<:MatrixTune
}
    "Check if adaption true in current iteration"
    adaption::A
    "θᵤ samples in current MCMC Phase, used for Σ estimation"
    chain::Matrix{T}
    "Posterior Covariance estimate"
    Σ::P
    "Cholesky decomposition of Inverse Posterior Covariance matrix Σ, s.t. Σ⁻¹ᶜʰᵒˡ*Σ⁻¹ᶜʰᵒˡ'=Σ⁻¹. Used to generate random draws in HMC/NUTS sampler."
    Σ⁻¹ᶜʰᵒˡ::C
    "Tuning parameter for Σ estimation"
    matrixtune::M
    function Proposal(
        type::Type{T}, adaption::A, tune::M, param_length::Int64, warmup_length::Int64
    ) where {T<:AbstractFloat,A<:BaytesCore.UpdateBool,M<:MatrixTune}
        Σ, Σ⁻¹ᶜʰᵒˡ = init(T, tune.metric, param_length)
        chain = zeros(T, param_length, warmup_length)
        return new{typeof(adaption),eltype(chain),typeof(Σ),typeof(Σ⁻¹ᶜʰᵒˡ),M}(
            adaption, chain, Σ, Σ⁻¹ᶜʰᵒˡ, tune
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Calculate regularized Covariance Matrix and Cholesky decomposition.

# Examples
```julia
```

"""
function update!(proposal::P, metric::MDense) where {P<:Proposal}
    @unpack Σ, Σ⁻¹ᶜʰᵒˡ, matrixtune, chain = proposal
    @unpack metric, shrinkage = matrixtune
    ## Compute new estimate
    Σ .= get_Σ(LinearAlgebra.Symmetric(cov(chain')), shrinkage)
    Σ⁻¹ᶜʰᵒˡ .= LinearAlgebra.cholesky(LinearAlgebra.inv(Σ)).L
    ## Pack container and return matrices
    @pack! proposal = Σ, Σ⁻¹ᶜʰᵒˡ
    return nothing
end
function update!(proposal::P, metric::MDiagonal) where {P<:Proposal}
    @unpack Σ, Σ⁻¹ᶜʰᵒˡ, matrixtune, chain = proposal
    @unpack metric, shrinkage = matrixtune
    ## Compute new estimate
    Σ .= get_Σ(LinearAlgebra.Diagonal(vec(var(chain; dims=2))), shrinkage)
    Σ⁻¹ᶜʰᵒˡ .= Diagonal(.√LinearAlgebra.inv.(LinearAlgebra.diag(Σ)))
    ## Pack container and return matrices
    @pack! proposal = Σ, Σ⁻¹ᶜʰᵒˡ
    return nothing
end
function update!(proposal::P, metric::MUnit) where {P<:Proposal}
    return nothing
end
update!(proposal::P) where {P<:Proposal} = update!(proposal, proposal.matrixtune.metric)

############################################################################################
"""
$(SIGNATURES)
Assign new chain buffer with dedicated size.

# Examples
```julia
```

"""
function chain!(
    proposal::P, phasename::N, Niterations::Integer
) where {P<:Proposal,N<:Union{Warmup,Adaptionˢˡᵒʷ}}
    proposal.chain = zeros(eltype(proposal.chain), size(proposal.chain, 1), Niterations)
    return nothing
end
function chain!(
    proposal::P, phasename::N, Niterations::Integer
) where {P<:Proposal,N<:Union{Adaptionᶠᵃˢᵗ}}#, Exploration}}
    #!NOTE: Do not create a new chain for Adaptionᶠᵃˢᵗ (before Exploration cycle), as in this phase Σ is kept constant. (But still have to calculate get_Σ!(proposal) for current cycle)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Update Proposal with new parameter θᵤ.

# Examples
```julia
```

"""
function update!(
    proposal::P,
    phaseupdate::Val{true},
    phasename::N,
    phase::PhaseTune,
    θᵤ::AbstractVector{T},
) where {P<:Proposal,N<:Union{Warmup,Adaptionˢˡᵒʷ,Adaptionᶠᵃˢᵗ},T<:Real}
    #NOTE!: Proposal will be updated at the beginning of Adaptionˢˡᵒʷ and Adaptionᶠᵃˢᵗ, but not at the beginning of Exploration (from Adaptionᶠᵃˢᵗ to Exploration, only local tuning parameter will be updated)
    ## Add current sample to chain
    @inbounds for iter in eachindex(θᵤ)
        proposal.chain[iter, phase.iter.current] = θᵤ[iter]
    end
    ## Estimate new covariance matrix
    update!(proposal)
    ## Preallocate new buffer for chain in next phase
    chain!(proposal, phasename, phase.iterations[phase.counter.current])
    return nothing
end
function update!(
    proposal::P,
    phaseupdate::Val{true},
    phasename::Exploration,
    phase::PhaseTune,
    θᵤ::AbstractVector{T},
) where {P<:Proposal,T<:Real}
    return nothing
end

function update!(
    proposal::P,
    phaseupdate::Val{false},
    phasename::N,
    phase::PhaseTune,
    θᵤ::AbstractVector{T},
) where {P<:Proposal,N<:Union{Warmup,Adaptionˢˡᵒʷ},T<:Real}
    ## Add current sample to chain
    @inbounds for iter in eachindex(θᵤ)
        proposal.chain[iter, phase.iter.current] = θᵤ[iter]
    end
    return nothing
end
function update!(
    proposal::P,
    phaseupdate::Val{false},
    phasename::N,
    phase::PhaseTune,
    θᵤ::AbstractVector{T},
) where {P<:Proposal,N<:Union{Adaptionᶠᵃˢᵗ,Exploration},T<:Real}
    #NOTE!: chain will be updated during Warmup and each Adaptionˢˡᵒʷ cycles, so can be used to estimate Σ for each Adaptionˢˡᵒʷ cycle and Adaptionᶠᵃˢᵗ cycle.
    #!NOTE: NO need to record during Adaptionᶠᵃˢᵗ, as during Exploration no Σ update.
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Update Proposal according to current tuning phase.

# Examples
```julia
```

"""
function update!(
    proposal::P, proposalupdate::BaytesCore.UpdateFalse, θᵤ, phasename, phase
) where {P<:Proposal}
    return nothing
end
function update!(
    proposal::P,
    proposalupdate::BaytesCore.UpdateTrue,
    θᵤ::AbstractVector{T},
    phasename::N,
    phase::PhaseTune,
) where {P<:Proposal,T<:Real,N<:Union{Warmup,Adaptionˢˡᵒʷ,Adaptionᶠᵃˢᵗ,Exploration}}
    ## Update Proposal chain and covariance estimate
    update!(proposal, Val(phase.update.current), phasename, phase, θᵤ)
    return nothing
end
function update!(
    proposal::P, θᵤ::AbstractVector{T}, phase::PhaseTune
) where {P<:Proposal,T<:Real}
    return update!(
        proposal, proposal.adaption, θᵤ, phase.name[phase.counter.current], phase
    )
end

############################################################################################
# Export
export MatrixMetric, MDense, MDiagonal, MUnit, ConfigProposal, MatrixTune, Proposal, update!
