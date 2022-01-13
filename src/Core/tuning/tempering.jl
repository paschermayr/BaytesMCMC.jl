#=
NOTES:
- Temperature at each iteration in Warmup and Adaptionˢˡᵒʷ.
- Temperature is kept constant in Adaptionᶠᵃˢᵗ, where final stepsize is tuned.
- Temperature is kept constant in Exploration.
=#
############################################################################################
"""
$(SIGNATURES)
Update Tempering with new temperature.

# Examples
```julia
```

"""
function update!(
    tempering::BaytesCore.TemperingTune,
    adaption::BaytesCore.UpdateTrue,
    phasename::N,
    iter::Integer
) where {N<:Union{Warmup,Adaptionˢˡᵒʷ}}
    ## Adapt tempering parameter
    update!(tempering.val, BaytesCore.update(tempering.parameter, iter))
    return nothing
end

function update!(
    tempering::BaytesCore.TemperingTune,
    adaption::BaytesCore.UpdateTrue,
    phasename::N,
    iter::Integer
) where {N<:Union{Adaptionᶠᵃˢᵗ, Exploration}}
    return nothing
end

function update!(
    tempering::BaytesCore.TemperingTune,
    adaption::BaytesCore.UpdateFalse,
    phasename::N,
    iter::Integer
) where {N}
    return nothing
end

############################################################################################
function update!(
    tempering::BaytesCore.TemperingTune, phase::PhaseTune, iter::Integer
)
    return update!(
        tempering, tempering.adaption, phase.name[phase.counter.current], iter
    )
end

############################################################################################
# Export
export update!
