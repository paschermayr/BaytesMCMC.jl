################################################################################
function init(::Type{Custom}, config::ConfigCustom, objective::Objective, difftune)
    return Custom(BaytesDiff.ℓDensityResult(objective), config.proposal)
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::Custom,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsCustom{model.info.reconstruct.default.output}
end

################################################################################
# Export
export infer, init
