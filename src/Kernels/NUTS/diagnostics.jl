############################################################################################
"""
$(TYPEDEF)
Diagnostics fields for NUTS sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsNUTS{T<:AbstractFloat} <: MCMCKernelDiagnostics
    "Log density (negative energy)."
    ℓH::T
    "Depth of the tree."
    depth::Int64
    "Reason for termination. See [`InvalidTree`](@ref) and [`REACHED_MAX_DEPTH`](@ref)."
    termination::InvalidTree
    "Acceptance rate statistic."
    acceptance_rate::T
    "Discretization size"
    ϵ::T
    "Number of leapfrog steps evaluated."
    steps::Int64
    "Directions for tree doubling (useful for debugging)."
    directions::Directions
    function DiagnosticsNUTS(
        ℓH::T,
        depth::Int64,
        termination::InvalidTree,
        acceptance_rate::T,
        ϵ::T,
        steps::Int64,
        directions::Directions,
    ) where {T<:AbstractFloat}
        @argcheck ϵ > 0.0 "Discretization size has to be positive"
        return new{T}(ℓH, depth, termination, acceptance_rate, ϵ, steps, directions)
    end
end

"""
$(SIGNATURES)
Summarize results of Vector of diagnostics.

# Examples
```julia
```

"""
function results(
    diagnostics::AbstractVector{DiagnosticsNUTS{F}}, Ndigits::Integer, quantiles::Vector{T}
) where {F<:AbstractFloat,T<:Real}
    println(
        "NUTS sampler had ",
        sum(is_divergent(diagnostics[iter].termination) for iter in eachindex(diagnostics)),
        " divergent transitions. Average (final) stepsize of ",
        round.(
            mean(diagnostics[iter].ϵ for iter in eachindex(diagnostics)); digits=Ndigits
        ),
        " (",
        round.(diagnostics[end].ϵ; digits=Ndigits),
        ") with ",
        round.(
            mean(diagnostics[iter].steps for iter in eachindex(diagnostics)); digits=Ndigits
        ),
        " (",
        diagnostics[end].steps,
        ") number of steps and depth of ",
        round.(
            mean(diagnostics[iter].depth for iter in eachindex(diagnostics)); digits=Ndigits
        ),
        " (",
        diagnostics[end].depth,
        ").",
    )
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Show relevant diagnostic results.

# Examples
```julia
```

"""
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsNUTS}
    return function showvalues()
        return (:NUTS, "diagnostics"),
        (:ϵ, diagnostics.ϵ), (:steps, diagnostics.steps),
        (:ℓH, diagnostics.ℓH)
    end
end

############################################################################################
# Export
export DiagnosticsNUTS, generate_showvalues
