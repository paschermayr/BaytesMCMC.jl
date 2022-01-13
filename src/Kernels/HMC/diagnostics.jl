############################################################################################
"""
$(TYPEDEF)
Diagnostics fields for HMC sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsHMC{T<:AbstractFloat} <: MCMCKernelDiagnostics
    "Discretization size"
    ϵ::T
    "Number of Leapfrog steps"
    steps::Int64
    function DiagnosticsHMC(ϵ::T, steps::Int64) where {T<:AbstractFloat}
        @argcheck steps > 0 "Minimum number of Leapfrog steps is 1."
        @argcheck ϵ > 0.0 "Discretization size has to be positive."
        return new{T}(ϵ, steps)
    end
end

function results(
    diagnostics::AbstractVector{DiagnosticsHMC{F}}, Ndigits::Integer, quantiles::Vector{T}
) where {F<:AbstractFloat,T<:Real}
    println(
        "HMC sampler has average (final) stepsize of ",
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
        ") number of steps.",
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
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsHMC}
    return function showvalues()
        return (:HMC, "diagnostics"), (:ϵ, diagnostics.ϵ), (:steps, diagnostics.steps)
    end
end

############################################################################################
#export
export DiagnosticsHMC, generate_showvalues
