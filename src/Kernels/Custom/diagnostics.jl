################################################################################
"""
$(TYPEDEF)
Default Configuration for Custom sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsCustom{T<:AbstractFloat} <: MCMCKernelDiagnostics
    "Discretization size"
    ϵ::T
    function DiagnosticsCustom(ϵ::T) where {T<:AbstractFloat}
        ArgCheck.@argcheck ϵ > 0.0 "Discretization size has to be positive"
        return new{T}(ϵ)
    end
end

function results(
    diagnostics::AbstractVector{DiagnosticsCustom{F}},
    Ndigits::Integer,
    quantiles::Vector{T},
) where {F<:AbstractFloat, T<:Real}
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
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsCustom}
    return function showvalues()
        return ((:Custom, "diagnostics"), )
    end
end

################################################################################
#export
export DiagnosticsCustom, generate_showvalues
