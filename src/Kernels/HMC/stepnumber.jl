############################################################################################
"""
$(TYPEDEF)
Contains information for number of discretization steps in MCMC algorithm.

# Fields
$(TYPEDFIELDS)
"""
mutable struct StepNumberTune{T<:BaytesCore.UpdateBool}
    "If true, number of steps will be adapted."
    adaption::T
    "Number of discretization steps."
    steps::Int64
    "Maximum number of discretization steps."
    stepsᵐᵃˣ::Int64
    "Desired Integration time"
    ∫dt::Float64
    function StepNumberTune(
        adaption::A, steps::Int64, stepsᵐᵃˣ::Int64, ∫dt::Float64
    ) where {A<:UpdateBool}
        return new{typeof(adaption)}(adaption, steps, stepsᵐᵃˣ, ∫dt)
    end
end

############################################################################################
function update!(
    number::StepNumberTune, adaption::BaytesCore.UpdateTrue, ϵ::F
) where {F<:AbstractFloat}
    @unpack stepsᵐᵃˣ, ∫dt = number
    #!NOTE:  max(10^-4, ϵ) instead of ϵ if (cornercase) ϵ so small that after division number would no longer be Int64 ~ 10^-5 more than enough for max steps reach
    stepsᵖ = max(1, Int(floor(∫dt / max(10^-3, ϵ)))) #At least 1 Leapfrog step, target Integration time of 1
    number.steps = min(stepsᵐᵃˣ, stepsᵖ)
    return nothing
end
function update!(
    number::StepNumberTune, adaption::BaytesCore.UpdateFalse, ϵ::F
) where {F<:AbstractFloat}
    return nothing
end
function update!(number::StepNumberTune, ϵ::F) where {F<:AbstractFloat}
    return update!(number, number.adaption, ϵ)
end

############################################################################################
#export
export StepNumberTune, update!
