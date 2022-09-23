############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing MCMC sampler easier in sampling library.

# Examples
```julia
```

"""
struct MCMCConstructor{M,S<:Union{Symbol,NTuple{k,Symbol} where k},D<:MCMCDefault} <: AbstractConstructor
    "Valid MCMC kernel."
    kernel::M
    "Parmeter to be tagged in MCMC sampler."
    sym::S
    "MCMC Default Arguments"
    default::D
    function MCMCConstructor(
        kernel::Type{M}, sym::S, default::D
    ) where {M<:MCMCKernel,S<:Union{Symbol,NTuple{k,Symbol} where k},D<:MCMCDefault}
        tup = BaytesCore.to_Tuple(sym)
        return new{typeof(kernel),typeof(tup),D}(kernel, tup, default)
    end
end
function (constructor::MCMCConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    temperature::F,
    info::BaytesCore.SampleDefault
) where {D, F<:AbstractFloat}
    return MCMC(
        _rng,
        constructor.kernel,
        Objective(model, data, Tagged(model, constructor.sym), temperature),
        constructor.default,
        info
    )
end
function MCMC(
    kernel::Type{M}, sym::S; kwargs...
) where {M<:MCMCKernel,S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return MCMCConstructor(kernel, sym, MCMCDefault(; kwargs...))
end

############################################################################################
function infer(diagnostics::Type{AbstractDiagnostics}, kernel::MCMCKernel)
    return println("No known diagnostics for given kernel")
end

"""
$(SIGNATURES)
Infer MCMC diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    mcmc::MCMC,
    model::ModelWrapper,
    data::D,
) where {D}
#    Tlik = eltype(mcmc.kernel.result.θᵤ)
    TKernel = infer(_rng, diagnostics, mcmc.kernel, model, data)
    TPrediction = infer(_rng, mcmc, model, data)
    TGenerated, TGenerated_algorithm = infer_generated(_rng, mcmc, model, data)
    return MCMCDiagnostics{TPrediction,TKernel,TGenerated, TGenerated_algorithm}
end

"""
$(SIGNATURES)
Infer type of predictions of MCMC sampler.

# Examples
```julia
```

"""
function infer(_rng::Random.AbstractRNG, mcmc::MCMC, model::ModelWrapper, data::D) where {D}
    objective = Objective(model, data, mcmc.tune.tagged)
    return typeof(predict(_rng, mcmc, objective))
end

"""
$(SIGNATURES)
Infer type of generated quantities of MCMC sampler.

# Examples
```julia
```

"""
function infer_generated(
    _rng::Random.AbstractRNG, mcmc::MCMC, model::ModelWrapper, data::D
) where {D}
    objective = Objective(model, data, mcmc.tune.tagged)
    TGenerated = typeof(generate(_rng, objective, mcmc.tune.generated))
    TGenerated_algorithm = typeof(generate(_rng, mcmc, objective, mcmc.tune.generated))
    return TGenerated, TGenerated_algorithm
end

############################################################################################
"""
$(SIGNATURES)
Generate statistics for algorithm given model parameter and data.

# Examples
```julia
```

"""
function generate(_rng::Random.AbstractRNG, algorithm::MCMC, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, algorithm::MCMC, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, algorithm, objective)
end
function generate(_rng::Random.AbstractRNG, algorithm::MCMC, objective::Objective, gen::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Print result for a single trace.

# Examples
```julia
```

"""
function results(
    diagnosticsᵛ::AbstractVector{M}, mcmc::MCMC, Ndigits::Integer, quantiles::Vector{T}
) where {T<:Real,M<:MCMCDiagnostics}
    println(
        "### ",
        Base.nameof(typeof(mcmc.kernel)),
        " parameter target: ",
        keys(mcmc.tune.tagged.parameter),
    )
    println(
        "Sampler finished after ",
        size(diagnosticsᵛ, 1),
        " iterations with acceptance rates of ",
        round.(
            mean(diagnosticsᵛ[iter].accept.accepted for iter in eachindex(diagnosticsᵛ)) *
            100;
            digits=Ndigits,
        ),
        "%.",
    )
    println(
        "Avg. initial ℓposterior: ",
        round(mean(diagnosticsᵛ[begin].base.ℓobjective); digits=Ndigits),
        ", Avg. final ℓposterior: ",
        round(mean(diagnosticsᵛ[end].base.ℓobjective); digits=Ndigits),
        ".",
    )
    ## Print kernel specific diagnostics
    results(
        [diagnosticsᵛ[iter].kernel for iter in eachindex(diagnosticsᵛ)], Ndigits, quantiles
    )
    ## Print Divergences
    print_divergences(diagnosticsᵛ, mcmc.tune.phase)
    return nothing
end

############################################################################################
function result!(mcmc::MCMC, result::L) where {L<:ℓObjectiveResult}
    mcmc.kernel.result = result
    return nothing
end

function get_result(mcmc::MCMC)
    return mcmc.kernel.result
end

function predict(_rng::Random.AbstractRNG, mcmc::MCMC, objective::Objective)
    return predict(_rng, objective)
end

############################################################################################
# Export
export MCMCConstructor, infer, predict
