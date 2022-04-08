############################################################################################
function init(
    ::Type{NUTS},
    config::ConfigNUTS,
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return NUTS(
        ModelWrappers.log_density_and_gradient(objective, difftune), difftune, config.energy
    )
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::NUTS,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsNUTS{model.info.reconstruct.default.output}
end

############################################################################################
# Export
export init, infer
