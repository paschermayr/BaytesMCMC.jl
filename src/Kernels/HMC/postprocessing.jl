############################################################################################
function init(
    ::Type{HMC},
    config::ConfigHMC,
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return HMC(
        ModelWrappers.log_density_and_gradient(objective, difftune),
        difftune,
        config.energy,
        StepNumberTune(
            config.stepnumberadaption, config.steps, config.maxsteps, config.∫dt
        ),
    )
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::HMC,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsHMC{model.info.flattendefault.output}
end
############################################################################################
#export
export init, infer