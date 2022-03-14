############################################################################################
"""
$(TYPEDEF)

MCMC Tuning container.

# Fields
$(TYPEDFIELDS)
"""
struct MCMCTune{
    A<:BaytesCore.UpdateBool,
    B<:BaytesCore.UpdateBool,
    F<:AbstractFloat,
    T<:Tagged,
    E<:Tuple,
    P<:Proposal} <: AbstractTune
    "Tagged Parameter."
    tagged::T
    "Current Phase in MCMC Cycle"
    phase::PhaseTune{E}
    "Stepsize container"
    stepsize::StepSizeTune{A,F}
    "Information for posterior covariance estimate"
    proposal::P
    "Boolean if generated quantities should be generated while sampling"
    generated::B
    "Current iteration number"
    iter::Iterator
    function MCMCTune(
        objective::Objective,
        phase::PhaseTune{E},
        stepsize::StepSizeTune{A,F},
        proposal::P,
        generate::B,
    ) where {A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool,F<:AbstractFloat,E<:Tuple,P<:Proposal}
        #!NOTE: Start with 0, so first proposal step will update iter to 1
        iter = Iterator(0)
        return new{A,B,F,typeof(objective.tagged),E,P}(
            objective.tagged, phase, stepsize, proposal, generate, iter
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Update MCMC tuning fields at current iteration.

# Examples
```julia
```

"""
function update!(
    tune::MCMCTune, result::S, acceptrate::T
) where {S<:ℓObjectiveResult,T<:Real}
    @unpack θᵤ = result
    ##  Update Current iteration counter
    update!(tune.iter)
    ## Update current sampling phase
    #!NOTE: Done AFTER tune.iter update; both iterations start from 0, so to check if phase needs to be updated, tune.iter has to be updated before
    update!(tune.phase, tune.iter.current)
    ## Update stepsize and proposal parameter given current sampling phase
    update!(tune.stepsize, acceptrate, tune.phase)
    update!(tune.proposal, θᵤ, tune.phase)
    ## Pack container
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Some basic tuning output figures. Useful for debugging.

# Examples
```julia
```

"""
function printtune(tune::MCMCTune, diagparam::Integer=size(tune.proposal.chain, 2))
    ArgCheck.@argcheck tune.iter.current > 0 "Need to run a proposal step before accessing tuning configuration."
    println("#################################################")
    println("Tune iter: ", tune.iter.current)
    println("Phase iter: ", tune.phase.iter.current)
    println("Phase name: ", tune.phase.name[tune.phase.counter.current])
    println("Phase update: ", tune.phase.update.current)
    println("Proposal update: ", tune.proposal.adaption)
    println("Phase counter: ", tune.phase.counter.current)
    #!NOTE: This will be Dualaveraging Parameter for first update! in new samplingphase, hence in final sampling phase only the smoothed ϵ is returned, and the dualaverage has no impact cause it wont compute anymore when Exploration+update=false
    #!NOTE: Hence, the stepsize multiplier will only come into effect at NEXT iteration with un-smoothed epsilon, in case adaption is still performed
    println("Phase stepsize: ", tune.stepsize.ϵ)
    println(
        "Phase (avg) stepsize: ",
        exp(tune.stepsize.dualaverage.logϵ),
        " (",
        exp(tune.stepsize.dualaverage.logϵ̄),
        ")",
    )
    println(
        "Proposal chain current row: ", tune.proposal.chain[end, tune.phase.iter.current]
    )
    println("Proposal chain last element: ", tune.proposal.chain[end])
    println("Proposal covariance: ", tune.proposal.Σ.diag[1:diagparam])
    return nothing
end

############################################################################################
# Export
export MCMCTune
