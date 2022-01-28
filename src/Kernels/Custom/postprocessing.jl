################################################################################
function init(::Type{Custom}, config::ConfigCustom, objective::Objective, difftune)
    return Custom(ModelWrappers.ℓDensityResult(objective), config.proposal)
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::Custom,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsCustom
end

################################################################################
# Export
export infer, init
