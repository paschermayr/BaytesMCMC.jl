################################################################################
function init(::Type{Metropolis}, config::ConfigMetropolis, objective::Objective, difftune)
    return Metropolis(BaytesDiff.ℓDensityResult(objective))
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::Metropolis,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsMetropolis{model.info.reconstruct.default.output}
end

################################################################################
# Export
export infer, init
