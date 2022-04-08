############################################################################################
function init(
    ::Type{MALA},
    config::ConfigMALA,
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return MALA(ModelWrappers.log_density_and_gradient(objective, difftune), difftune)
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::MALA,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsMALA{model.info.reconstruct.default.output}
end

############################################################################################
# Export
export init, infer
