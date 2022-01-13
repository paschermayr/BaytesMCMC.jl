################################################################################
function init(::Type{Metropolis}, config::ConfigMetropolis, objective::Objective, difftune)
    return Metropolis(ModelWrappers.ℓDensityResult(objective))
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::Metropolis,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsMetropolis{model.info.flattendefault.output}
end

################################################################################
# Export
export infer, init
