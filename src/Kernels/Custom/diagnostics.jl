################################################################################
"""
$(TYPEDEF)
Default Configuration for Custom sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsCustom <: MCMCKernelDiagnostics
    function DiagnosticsCustom()
        return new()
    end
end

function results(
    diagnostics::AbstractVector{DiagnosticsCustom},
    Ndigits::Integer,
    quantiles::Vector{T},
) where {T<:Real}
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
