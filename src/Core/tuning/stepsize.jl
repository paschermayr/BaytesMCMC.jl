#=
NOTES:
- Stepsize will be updated in each iteration in phases Warmup, Adaptionˢˡᵒʷ, Adaptionᶠᵃˢᵗ.
- Within phases updates are the un-smoothed stepsizes.
- At the end of each iteration, the smoothed stepsize is returned. Hence, in Exploration
    the smoothed stepsize from phase Adaptionᶠᵃˢᵗ is used and kept constant.
- After the first proposal step in new iteration in phases Warmup, Adaptionˢˡᵒʷ, Adaptionᶠᵃˢᵗ,
    DualAverageParameter is updated and stepsize multiplier will only come into effect at
    NEXT iteration with un-smoothed epsilon, in case adaption is still performed.
=#

############################################################################################
"""
$(TYPEDEF)

Default configuration for stepsize adaption.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigStepsize{A<:BaytesCore.UpdateBool,B<:BaytesCore.UpdateBool}
    "Initial Discretization size"
    ϵ::Float64
    "Step size adaption"
    stepsizeadaption::A
    "Boolean if initial stepsize should be estimated"
    initialstepsize::B
    function ConfigStepsize(;
        ϵ = 0.1,
        stepsizeadaption = BaytesCore.UpdateTrue(),
        initialstepsize = BaytesCore.UpdateFalse()
        )
        @argcheck ϵ > 0.0 "Discretization size has to be positive"
        return new{typeof(stepsizeadaption), typeof(initialstepsize)}(
            ϵ, stepsizeadaption, initialstepsize
            )
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains information for default Dual Averaging algorithm.

# Fields
$(TYPEDFIELDS)
"""
struct DualAverageParameter{T<:AbstractFloat}
    "Target acceptance rate"
    δ::T
    "Regularization scale"
    γ::T
    "Relaxation exponent - for Average log step size"
    κ::T
    "Offset"
    t₀::Int64
    function DualAverageParameter(δ::T, γ::T, κ::T, t₀::Int64) where {T<:AbstractFloat}
        @argcheck 0 < δ < 1
        @argcheck γ > 0
        @argcheck 0.5 < κ ≤ 1
        @argcheck t₀ ≥ 0
        return new{T}(δ, γ, κ, t₀)
    end
end
function DualAverageParameter(
    ::Type{T}; δ=0.80, γ=0.05, κ=0.75, t₀=10
) where {T<:AbstractFloat}
    return DualAverageParameter(T(δ), T(γ), T(κ), t₀)
end

############################################################################################
"""
$(TYPEDEF)

Contains DualAverage tuning information and runtime parameter.

# Fields
$(TYPEDFIELDS)
"""
struct DualAverage{T<:AbstractFloat}
    #!NOTE: Updated each iteration
    adaption::DualAverageParameter{T}
    "Upwards bias for target acceptance rate - proposals are biased upwards to stay away from 0."
    μ::T
    "Time update, starts with 0"
    t::Int64
    "Average part of first equation in Hoffman(2014), p 1607, (6)"
    H̄::T
    "Log step"
    logϵ::T
    "AVERAGED log step"
    logϵ̄::T
    function DualAverage(
        adaption::DualAverageParameter{T}, μ::A, t::I, H̄::B, logϵ::C, logϵ̄::D
    ) where {T<:Real,I<:Integer, A<:Real, B<:Real, C<:Real, D<:Real}
        return new{T}(adaption, μ, t, H̄, logϵ, logϵ̄)
    end
end

############################################################################################
"""
$(SIGNATURES)
Initialize new DualAverage struct.

# Examples
```julia
```

"""
function init(::Type{DualAverage}, δ, ϵ::T; multiplier=T(10.0)) where {T<:AbstractFloat}
    @argcheck ϵ > 0
    logϵ = log(ϵ)
    return DualAverage(
        DualAverageParameter(T; δ=δ),
        log(multiplier) + logϵ,
        0,
        zero(logϵ),
        logϵ,
        zero(logϵ),
    )
end

############################################################################################
"""
$(SIGNATURES)
Update DualAverage struct.

# Examples
```julia
```

"""
function update(dualaverage::DualAverage, acceptratio::T) where {T<:AbstractFloat}
    @argcheck 0 ≤ acceptratio ≤ 1
    @unpack adaption, μ, t, H̄, logϵ, logϵ̄ = dualaverage
    @unpack δ, γ, κ, t₀ = adaption
    ## Update t and H
    t += 1
    H̄ += (δ - acceptratio - H̄) / (t + t₀)
    ## Calculate logϵ and MA
    logϵ = μ - T(√t) / γ * H̄
    logϵ̄ += t^(-κ) * (logϵ - logϵ̄)
    ## Pack container
    return DualAverage(adaption, μ, t, H̄, logϵ, logϵ̄) #T(logϵ), T(logϵ̄ ))
end

############################################################################################
"""
$(TYPEDEF)

Initial Step Size Search factors.

# Fields
$(TYPEDFIELDS)
"""
struct InitialStepsizeSearch{T<:AbstractFloat}
    "Lowest local acceptance rate."
    a_min::T
    "Highest local acceptance rate."
    a_max::T
    "Initial stepsize."
    ϵ₀::T
    "Scale factor for initial bracketing, > 1."
    C::T
    "Maximum number of iterations for initial bracketing."
    maxiter_crossing::Int64
    "Maximum number of iterations for bisection."
    maxiter_bisect::Int64
    function InitialStepsizeSearch(
        ::Type{T};
        a_min=0.25,
        a_max=0.75,
        ϵ₀=0.5,
        C=2.0,
        maxiter_crossing=1000,
        maxiter_bisect=1000,
    ) where {T}
        @argcheck 0 < a_min < a_max < 1
        @argcheck 0 < ϵ₀
        @argcheck 1 < C
        @argcheck maxiter_crossing ≥ 50
        @argcheck maxiter_bisect ≥ 50
        return new{T}(a_min, a_max, ϵ₀, C, maxiter_crossing, maxiter_bisect)
    end
end

############################################################################################
"""
$(SIGNATURES)
Find the stepsize for which the local acceptance rate `A(ϵ)` crosses `a`.

# Examples
```julia
```

"""
function find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀=A(ϵ₀))
    @unpack a_min, a_max, C, maxiter_crossing = parameters
    s, a = Aϵ₀ > a_max ? (1.0, a_max) : (-1.0, a_min)
    if s < 0                    # when A(ϵ) < a,
        C = 1 / C                 # decrease ϵ
    end
    for _ in Base.OneTo(maxiter_crossing)
        ϵ = ϵ₀ * C
        Aϵ = A(ϵ)
        if s * (Aϵ - a) ≤ 0
            return ϵ₀, Aϵ₀, ϵ, Aϵ
        else
            ϵ₀ = ϵ
            Aϵ₀ = Aϵ
        end
    end
    dir = s > 0 ? "below" : "above"
    return error("Reached maximum number of iterations searching for ϵ from $(dir).")
end

############################################################################################
"""
$(SIGNATURES)
Return the desired stepsize `ϵ` by bisection.

# Examples
```julia
```

"""
function bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀=A(ϵ₀), Aϵ₁=A(ϵ₁))
    @unpack a_min, a_max, maxiter_bisect = parameters
    @argcheck ϵ₀ < ϵ₁
    @argcheck Aϵ₀ > a_max && Aϵ₁ < a_min
    for _ in Base.OneTo(maxiter_bisect)
        ϵₘ = Statistics.middle(ϵ₀, ϵ₁) #0.5*(ϵ₀ + ϵ₁)
        Aϵₘ = A(ϵₘ)
        if a_min ≤ Aϵₘ ≤ a_max  # in
            return ϵₘ
        elseif Aϵₘ < a_min      # above
            ϵ₁ = ϵₘ
            Aϵ₁ = Aϵₘ
        else                    # below
            ϵ₀ = ϵₘ
            Aϵ₀ = Aϵₘ
        end
    end
    return error("Reached maximum number of iterations while bisecting interval for ϵ.")
end

############################################################################################
"""
$(SIGNATURES)
Find an initial stepsize that matches the conditions of `parameters` (see
[`InitialStepsizeSearch`](@ref)). `A` is the local acceptance ratio (unbounded).

# Examples
```julia
```

"""
function find_initial_stepsize(parameters::InitialStepsizeSearch, A)
    @unpack a_min, a_max, ϵ₀ = parameters
    #!NOTE: A is a closure for the MCMC step to calculate acceptance rate for 1 MCMC step
    Aϵ₀ = A(ϵ₀)
    if a_min ≤ Aϵ₀ ≤ a_max
        ϵ₀
    else
        ϵ₀, Aϵ₀, ϵ₁, Aϵ₁ = find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀)
        #!NOTE: in interval
        if a_min ≤ Aϵ₁ ≤ a_max
            ϵ₁
            #!NOTE: order as necessary
        elseif ϵ₀ < ϵ₁
            bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀, Aϵ₁)
        else
            bisect_stepsize(parameters, A, ϵ₁, ϵ₀, Aϵ₁, Aϵ₀)
        end
    end
end

############################################################################################
"""
$(TYPEDEF)

Tuning stepsize parameter for MCMC algorithm.

# Fields
$(TYPEDFIELDS)
"""
mutable struct StepSizeTune{A<:BaytesCore.UpdateBool,F<:AbstractFloat}
    "If true, stepsize will be adapted."
    adaption::A
    #!NOTE: isbits(dualaverage) == true
    "Dualaverage struct"
    dualaverage::DualAverage{F}
    "Current stepsize"
    ϵ::F
    function StepSizeTune(
        adaption::A, dualaverage::DualAverage{F}, ϵ::F
    ) where {A<:BaytesCore.UpdateBool,F<:AbstractFloat}
        @argcheck 0.0 < ϵ
        return new{A,F}(adaption, dualaverage, ϵ)
    end
end

"""
$(SIGNATURES)
Update stepsize based on acceptance ratio α.

# Examples
```julia
```

"""
function update!(
    stepsize::StepSizeTune,
    stepsizeadaption::BaytesCore.UpdateTrue,
    α::T,
    samplingphase::N,
    iterationupdate::Val{false},
) where {N<:Union{Warmup,Adaptionˢˡᵒʷ,Adaptionᶠᵃˢᵗ},T<:AbstractFloat}
    @unpack dualaverage = stepsize
    ## Update stepsize
    dualaverage = update(dualaverage, α)
    ϵ = exp(dualaverage.logϵ)
    @pack! stepsize = dualaverage, ϵ
    return nothing
end
function update!(
    stepsize::StepSizeTune,
    stepsizeadaption::BaytesCore.UpdateTrue,
    α::T,
    samplingphase::Exploration,
    iterationupdate::Val{false},
) where {T<:AbstractFloat}
    return nothing
end
function update!(
    stepsize::StepSizeTune,
    stepsizeadaption::BaytesCore.UpdateTrue,
    α::T,
    samplingphase::S,
    iterationupdate::Val{true},
) where {S<:SamplingPhase,T<:AbstractFloat}
    @unpack dualaverage = stepsize
    ## Update stepsize
    dualaverage = update(dualaverage, α)
    ϵ = exp(dualaverage.logϵ̄) #!NOTE: SMOOTHED VERSION
    ## Assign new dualaverage parameter with smoothed log epsilon
    #= This will be Dualaveraging Parameter for first update! in new samplingphase, hence in final sampling phase only the smoothed ϵ is returned,
    and dualaverage has no impact cause it wont compute anymore when Exploration+update=false. Hence, the stepsize multiplier will only come
    into effect at NEXT iteration with un-smoothed epsilon, in case adaption is still performed. =#
    dualaverage = init(DualAverage, dualaverage.adaption.δ, ϵ)
    @pack! stepsize = dualaverage, ϵ
    return nothing
end
function update!(
    stepsize::StepSizeTune, stepsizeadaption::BaytesCore.UpdateFalse, α, samplingphase, iterationupdate
)
    return nothing
end
function update!(stepsize::StepSizeTune, α::T, phase::PhaseTune) where {T<:AbstractFloat}
    return update!(
        stepsize,
        stepsize.adaption,
        α,
        phase.name[phase.counter.current],
        Val(phase.update.current),
    )
end

############################################################################################
# Export
export StepSizeTune, ConfigStepsize, DualAverage, DualAverageParameter
