############################################################################################
# MALA specific diagnostics
"""
$(TYPEDEF)
Diagnostics for MALA sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsMALA{T<:AbstractFloat} <: MCMCKernelDiagnostics
    "Discretization size"
    ϵ::T
    function DiagnosticsMALA(ϵ::T) where {T<:AbstractFloat}
        @argcheck ϵ > 0.0 "Discretization size has to be positive"
        return new{T}(ϵ)
    end
end

function results(
    diagnostics::AbstractVector{DiagnosticsMALA{F}}, Ndigits::Integer, quantiles::Vector{T}
) where {F<:AbstractFloat,T<:Real}
    println(
        "MALA sampler has average (final) stepsize of ",
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
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsMALA}
    return function showvalues()
        return (:MALA, "diagnostics"), (:ϵ, diagnostics.ϵ)
    end
end

############################################################################################
#export
export DiagnosticsMALA, generate_showvalues
