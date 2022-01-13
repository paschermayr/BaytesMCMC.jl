################################################################################
"""
$(TYPEDEF)
Default Configuration for Metropolis sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsMetropolis{T<:AbstractFloat} <: MCMCKernelDiagnostics
    "Discretization size"
    ϵ::T
    function DiagnosticsMetropolis(ϵ::T) where {T<:AbstractFloat}
        ArgCheck.@argcheck ϵ > 0.0 "Discretization size has to be positive"
        return new{T}(ϵ)
    end
end

function results(
    diagnostics::AbstractVector{DiagnosticsMetropolis{F}},
    Ndigits::Integer,
    quantiles::Vector{T},
) where {F<:AbstractFloat,T<:Real}
    println(
        "Metropolis sampler has average (final) stepsize of ",
        round.(
            mean(diagnostics[iter].ϵ for iter in eachindex(diagnostics)); digits=Ndigits
        ),
        " (",
        round.(diagnostics[end].ϵ; digits=Ndigits),
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
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsMetropolis}
    return function showvalues()
        return (:Metropolis, "diagnostics"), (:ϵ, diagnostics.ϵ)
    end
end

################################################################################
#export
export DiagnosticsMetropolis, generate_showvalues
