############################################################################################
"""
$(TYPEDEF)

Sampling Container that determines tuning updates.

# Fields
$(TYPEDFIELDS)
"""
abstract type SamplingPhase end
" Find/converges to the typical set (estimators suffer from initial but ultimately transient biases)"
struct Warmup <: SamplingPhase end
"Explore typical set and tune MCMC parameter(initial bias rapidly vanishes and the estimators become much more accurate.)"
struct Adaptionˢˡᵒʷ <: SamplingPhase end
"Tune final stepsize for fixed Covariance Matrix."
struct Adaptionᶠᵃˢᵗ <: SamplingPhase end
"Sample parameter with tuned MCMC (gradually reducing the precision error of the MCMC estimators towards zero.)"
struct Exploration <: SamplingPhase end

############################################################################################
"""
$(TYPEDEF)

Information about current SamplingPhase.

# Fields
$(TYPEDFIELDS)
"""
struct PhaseTune{T<:Tuple}
    "Boolean if current iteration needs update."
    update::Updater
    "Counts current iteration in phase."
    iter::Iterator
    "MCMC Phases ~ counter = current cyle in slices/name/iterations."
    counter::Iterator
    "Vector of MCMC iterations at each phase."
    slices::Vector{Int64}
    "Name of Sampling phases."
    name::T
    "Counts total iterations."
    iterations::Vector{Int64}
    ## MCMC Windows
    "Counts cycles in adaption phases Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(),
    i.e., 1-5-1 means 1 window init, 5 adapt, 1 exploration."
    window::Vector{Int64}
    "Counts iteration in each cycle for adaption phases Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(),
    i.e., 50-25-50 means 1 time 50, 5 times 25*i, i in 1:5, 1 time 50."
    buffer::Vector{Int64}
    function PhaseTune(;
        window=[1, 5, 1],
        buffer=[50, 25, 50],
        phasenames=[Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(), Exploration()],
    )
        ArgCheck.@argcheck length(window) == length(buffer) == (length(phasenames) - 1) "Size of window, buffer, or phasenames are not matching."
        ## Initiate mutable update container
        update = Updater(false)
        #!NOTE: start with 0, so first propose!() call will update iter.current to 1, and proposal will be first filled with 1st column.
        iter = Iterator(0)
        #!NOTE: start with 1 as Warmup matrix is initiated before propose! steps
        counter = Iterator(1)
        ## Initiate Warmup and Adaption Cycles
        phase_name = SamplingPhase[]
        phase_iterations = Int64[]
        for i in eachindex(window)
            for j in Base.OneTo(window[i])
                push!(phase_iterations, convert(Int64, j * buffer[i]))
                push!(phase_name, phasenames[i])
            end
        end
        ## Include Exploration Cycle
        push!(phase_name, phasenames[end])
        push!(phase_iterations, 1)
        ## Calculate cumsum of cylces for chainₜ iterations
        phase_slices = cumsum(phase_iterations)
        ## Change type of phase_name for concrete struct type
        phase_name_tuple = Tuple(phase_name)
        return new{typeof(phase_name_tuple)}(
            update,
            iter,
            counter,
            phase_slices,
            phase_name_tuple,
            phase_iterations,
            window,
            buffer,
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Update current MCMC Phase.

# Examples
```julia
```

"""
function update!(phase::PhaseTune, iter::Int64)
    @unpack slices = phase
    ## Check if updated last iteration
    if phase.update.current
        ## If update == true, set updating to false if updated last iteration
        update!(phase.update)
        ## Set back to 0 and add 1 afterwards
        phase.iter.current = 0
    end
    ## If update now, change Sampling phase
    if phase.counter.current < length(slices) && slices[phase.counter.current] == iter
        update!(phase.update)
        update!(phase.counter)
    end
    ## Update current iteration
    update!(phase.iter)
    return nothing
end

############################################################################################
# Export
export SamplingPhase, Warmup, Adaptionˢˡᵒʷ, Adaptionᶠᵃˢᵗ, Exploration, PhaseTune
