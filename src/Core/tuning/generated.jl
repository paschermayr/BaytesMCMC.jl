############################################################################################
function generate(_rng::Random.AbstractRNG, objective::Objective, gen::Val{true})
    return ModelWrappers.generate(_rng, objective)
end
function generate(_rng::Random.AbstractRNG, objective::Objective, gen::Val{false})
    return nothing
end
############################################################################################
# Export
#export
