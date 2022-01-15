############################################################################################
"""
$(TYPEDEF)

MCMC Diagnostics container.

# Fields
$(TYPEDFIELDS)
"""
struct MCMCDiagnostics{R<:AbstractFloat,E<:MCMCKernelDiagnostics,T,G} <: AbstractDiagnostics
    ## Common MCMC output statistics
    "Evaluation of target function at current iteration."
    ℓθᵤ::R
    "Temperature for log posterior evaluation"
    temperature::R
    "Boolean if diverged."
    divergence::Bool
    "Acceptance Rate of current step."
    accept::AcceptStatistic{R}
    "Sampler specific diagnostics."
    sampler::E
    "Predicted sample of model."
    prediction::T
    "Generated quantities specified for objective"
    generated::G
    "Current iteration number."
    iter::Int64
    function MCMCDiagnostics(
        ℓθᵤ::R,
        temperature::R,
        divergence::Bool,
        accept::AcceptStatistic{R},
        kerneldiagnostics::E,
        prediction::T,
        generated::G,
        iter::Int64,
    ) where {R<:AbstractFloat,E<:MCMCKernelDiagnostics,T,G}
        return new{R,E,T,G}(
            ℓθᵤ, temperature, divergence, accept, kerneldiagnostics, prediction, generated, iter
        )
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
    for iter in 1:(length(counter) - 1)
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
export MCMCDiagnostics
