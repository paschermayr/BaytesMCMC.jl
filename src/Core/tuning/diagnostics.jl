############################################################################################
"""
$(TYPEDEF)

MCMC Diagnostics container.

# Fields
$(TYPEDFIELDS)
"""
struct MCMCDiagnostics{
    P,
    K<:MCMCKernelDiagnostics,
    G,
    A
} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{P}
    "Kernel specific diagnostics."
    kernel::K
    "Boolean if diverged."
    divergence::Bool
    "Acceptance Rate of current step."
    accept::AcceptStatistic
    "Generated quantities specified for objective"
    generated::G
    "Generated quantities specified for algorithm"
    generated_algorithm::A
    function MCMCDiagnostics(
        base::BaytesCore.BaseDiagnostics{P},
        kerneldiagnostics::K,
        divergence::Bool,
        accept::AcceptStatistic,
        generated::G,
        generated_algorithm::A
    ) where {P, K<:MCMCKernelDiagnostics, G, A}
        return new{P,K,G,A}(
            base, kerneldiagnostics, divergence, accept, generated, generated_algorithm
        )
    end
end

############################################################################################
function generate_showvalues(diagnostics::D) where {D<:MCMCDiagnostics}
    kernel = generate_showvalues(diagnostics.kernel)
    return function showvalues()
        return (:mcmc, "diagnostics"),
        (:iter, diagnostics.base.iter),
        (:logobjective, diagnostics.base.ℓobjective),
        (:Temperature, diagnostics.base.temperature),
        (:accepted, diagnostics.accept.accepted),
        (:acceptancerate, diagnostics.accept.rate),
        (:generated, diagnostics.generated),
        (:generated_algorithm, diagnostics.generated_algorithm),
        kernel()...
    end
end

############################################################################################
"""
$(SIGNATURES)
Print all divergences from a vector of MCMC Diagnostics.

# Examples
```julia
```

"""
function print_divergences(
    diagnosticsᵛ::AbstractVector{M}, phase::PhaseTune
) where {M<:MCMCDiagnostics}
    ## Assign phase names and ending indices
    phase_names = unique(phase.name)
    phase_sequence = cumsum(phase.window)
    phase_iterations = [phase.slices[iter] for iter in phase_sequence]
    push!(phase_iterations, max(size(diagnosticsᵛ, 1), phase_iterations[end] + 1))
    counter = zeros(Int64, length(phase_names))
    ## Loop through all chains and iterations to check for divergences
    for iter in eachindex(diagnosticsᵛ)
        if diagnosticsᵛ[iter].divergence
            _, phaseindex = findmax(iter .<= phase_iterations)
            counter[phaseindex] += 1
        end
    end
    ## Print results
    print("Divergences in: ")
    for iter in Base.OneTo(length(counter) - 1)
        print(Base.nameof(typeof(phase_names[iter])), ": ", counter[iter], ", ")
    end
    for iter in length(counter)
        println(Base.nameof(typeof(phase_names[iter])), ": ", counter[iter], ".")
    end
    ## Return divergences
    return phase_names, counter
end

############################################################################################
# Export
export MCMCDiagnostics, generate_showvalues
