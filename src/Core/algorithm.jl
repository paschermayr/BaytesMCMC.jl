############################################################################################
"""
$(TYPEDEF)

Default arguments for MCMC constructor.

# Fields
$(TYPEDFIELDS)
"""
struct MCMCDefault{K<:NamedTuple, S<:ConfigStepsize, P<:ConfigProposal, G, I<:ModelWrappers.AbstractInitialization, U<:BaytesCore.UpdateBool}
    "Individual keyword arguments for tuning different MCMC kernels."
    kernel::K
    "Stepsize default configuration for adaption."
    stepsize::S
    "Proposal distribution default configuration for adaption."
    proposal::P
    "Gradient backend used in MCMC step. Not used if Metropolis sampler is chosen."
    GradientBackend::G
    "Method to obtain initial Modelparameter, see 'AbstractInitialization'."
    init::I
    "Boolean if generate(_rng, objective) for corresponding model is stored in MCMC Diagnostics."
    generated::U
    function MCMCDefault(;
        kernel=(;),
        stepsize = ConfigStepsize(),
        proposal = ConfigProposal(),
        GradientBackend=:ForwardDiff,
        init=ModelWrappers.NoInitialization(),
        generated=BaytesCore.UpdateFalse()
    )
        ArgCheck.@argcheck (
            isa(GradientBackend, Symbol) || isa(GradientBackend, AnalyticalDiffTune)
        ) "GradientBackend keywords has to be either an AD symbol (:ForwardDiff, :ReverseDiff, :ReverseDiffUntaped, :Zyogte), or an AnalyticalDiffTune object."
        return new{
            typeof(kernel), typeof(stepsize), typeof(proposal), typeof(GradientBackend), typeof(init), typeof(generated)
        }(
            kernel, stepsize, proposal, GradientBackend, init, generated
        )
    end
end

############################################################################################
"""
$(TYPEDEF)

Stores information for proposal step.

# Fields
$(TYPEDFIELDS)
"""
struct MCMC{M<:MCMCKernel,N<:MCMCTune} <: AbstractAlgorithm
    "MCMC sampler"
    kernel::M
    "Tuning configuration for kernel."
    tune::N
    function MCMC(kernel::M, tune::N) where {M<:MCMCKernel,N<:MCMCTune}
        return new{M,N}(kernel, tune)
    end
end

############################################################################################
function MCMC(
    _rng::Random.AbstractRNG,
    kernel::Type{M},
    objective::Objective,
    default::MCMCDefault=MCMCDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
) where {M<:MCMCKernel}
    ## Checks before algorithm is initiated
    @unpack output = objective.model.info.reconstruct.default
    @unpack GradientBackend, generated = default
    ## Obtain initial parameter
    default.init(_rng, kernel, objective)
    @unpack model, data, tagged = objective
    ## Initialize default configuration for chosen algorithm
    config = init(AbstractConfiguration, kernel, objective, default.proposal; default.kernel...)
    ##	If a valid AD backend is provided, change it to an AutomaticDifftune Object
    if isa(GradientBackend, Symbol)
        GradientBackend = AutomaticDiffTune(GradientBackend, objective)
    end
    ## Initiate MCMC Algorithm
    mcmc = init(kernel, config, objective, GradientBackend)
    ## Initial Phase tune
    phasetune = PhaseTune(;
        window=config.window.window, buffer=config.window.buffer, phasenames=config.window.phasenames
    )
    ## Initial Proposal tune
    proposal = Proposal(
        output,
        default.proposal.proposaladaption,
        MatrixTune(default.proposal.metric, default.proposal.shrinkage),
        length(objective.tagged),
        phasetune.iterations[phasetune.counter.current],
        default.proposal.covariance
    )
    ## Tune stepsize in case it is not assigned via local acceptance rate bisection
    if default.stepsize.initialstepsize isa BaytesCore.UpdateTrue
        acceptrateᵗᵉᵐᵖ = get_acceptrate(_rng, mcmc, objective, proposal.Σ) # Functor to calculate local accpetance rate
        ϵ = find_initial_stepsize(
            InitialStepsizeSearch(output), acceptrateᵗᵉᵐᵖ
        )
    else
        ϵ = output(default.stepsize.ϵ)
    end
    stepsize = StepSizeTune(default.stepsize.stepsizeadaption, init(DualAverage, config.δ, ϵ), ϵ)
    ## Initial MCMC Tune struct, and return MCMC container
    mcmctune = MCMCTune(objective, phasetune, stepsize, proposal, generated)
    ## Return MCMC container
    return MCMC(mcmc, mcmctune)
end

############################################################################################
"""
$(SIGNATURES)
Propose new parameter with mcmc sampler. If update=true, objective function will be updated with input model and data.

# Examples
```julia
```

"""
function propose(_rng::Random.AbstractRNG, mcmc::MCMC, objective::Objective)
    #!NOTE: Temperature is fixed for propose() step and will not be adjusted
    ## Make MCMC Proposal step
    resultᵖ, divergent, accept, kernel_diagnostics = propagate(
        _rng, mcmc.kernel, mcmc.tune, objective
    )
    ## If accepted, update kernel and Model parameter
    if accept.accepted
        mcmc.kernel.result = resultᵖ
        ModelWrappers.unflatten_constrain!(objective.model, mcmc.tune.tagged, resultᵖ.θᵤ)
    end
    ## Upate tuning container - includes storing current parameter, updating sampling phase, discretization steps, and proposal distribution (in that order)
    update!(mcmc.tune, mcmc.kernel.result, accept.rate)
    diagnostics = MCMCDiagnostics(
        BaytesCore.BaseDiagnostics(
            mcmc.kernel.result.ℓθᵤ,
            objective.temperature,
            ModelWrappers.predict(_rng, mcmc, objective),
            mcmc.tune.iter.current
        ),
        kernel_diagnostics,
        divergent,
        accept,
        ModelWrappers.generate(_rng, objective, mcmc.tune.generated)
    )
    return objective.model.val, diagnostics
end

############################################################################################
"""
$(SIGNATURES)
Inplace version of propose.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    mcmc::MCMC,
    model::ModelWrapper,
    data::D,
    temperature::F = model.info.reconstruct.default.output(1.0),
    update::U=BaytesCore.UpdateTrue(),
) where {D,F<:AbstractFloat, U<:BaytesCore.UpdateBool}
    ## Update Objective with new model parameter from other MCMC samplers and/or new/latent data
    objective = Objective(model, data, mcmc.tune.tagged, temperature)
    update!(mcmc.kernel, objective, update) #Update Kernel with current objective/configs
    ## Compute MCMC step
    val, diagnostics = propose(_rng, mcmc, objective)
    ## If accepted, update kernel and Model parameter
    if diagnostics.accept.accepted
        model.val = val
    end
    return val, diagnostics
end

############################################################################################
# Export
export MCMC, MCMCDefault, propose, propose!
