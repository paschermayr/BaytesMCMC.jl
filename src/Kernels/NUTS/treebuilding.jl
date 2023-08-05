############################################################################################
"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct TrajectoryNUTS{S<:Hamiltonian,F,C}
    "Hamiltonian"
    H::S
    "Log density of negative log energy at initial point."
    ℓH₀::F
    "Stepsize for leapfrog."
    ϵ::F
    "Maximum tree depth."
    max_depth::Int64
    "Smallest decrease allowed in the log density."
    min_Δ::Float64
    "Turn statistic configuration."
    turn_statistic_configuration::C
    function TrajectoryNUTS(
        H::S,
        ℓH₀::F,
        ϵ::F,
        max_depth::Int64, # = DEFAULT_MAX_TREE_DEPTH,
        min_Δ=-1000.0,
        turn_statistic_configuration=Val{:generalized}(),
    ) where {S<:Hamiltonian,F<:AbstractFloat}
        @argcheck 0 < max_depth ≤ MAX_DIRECTIONS_DEPTH
        @argcheck min_Δ < 0
        C = typeof(turn_statistic_configuration)
        return new{S,F,C}(H, ℓH₀, ϵ, max_depth, min_Δ, turn_statistic_configuration)
    end
end

"""
$(SIGNATURES)
Move along the trajectory in the specified direction. Return the new position.

# Examples
```julia
```

"""
function move(trajectory::T, phasepoint::PhasePoint, fwd::Bool) where {T<:TrajectoryNUTS}
    @unpack H, ϵ = trajectory
    return leapfrog(H, phasepoint, fwd ? ϵ : -ϵ)
end

############################################################################################
# Proposal
"""
$(SIGNATURES)
Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of drawing a sample from the second (`logprob2`).
When `bias`, biases towards the second argument, introducing anti-correlations.

# Examples
```julia
```

"""
function biased_progressive_logprob2(
    bias::Bool, ω₁::T, ω₂::T, ω=BaytesCore.logaddexp(ω₁, ω₂)
) where {T<:Real}
    return ω₂ - (bias ? ω₁ : ω)
end

"""
$(SIGNATURES)
Random boolean which is `true` with the given probability `exp(logprob)`, which can be `≥ 1`
in which case no random value is drawn.

# Examples
```julia
```

"""
function rand_bool_logprob(_rng::Random.AbstractRNG, logprob::T) where {T<:Real}
    return logprob ≥ 0 || (randexp(_rng, T) > -logprob)
end

"""
$(SIGNATURES)
Calculate the log probability if selecting the subtree corresponding to `ω₂`. Being the log
of a probability, it is always `≤ 0`, but implementations are allowed to return and accept
values `> 0` and treat them as `0`. When `is_doubling`, the tree corresponding to `ω₂` was obtained from a doubling step (this
can be relevant eg for biased progressive sampling). The value `ω = logaddexp(ω₁, ω₂)` is provided for avoiding redundant calculations.
See [`biased_progressive_logprob2`](@ref) for an implementation.

# Examples
```julia
```

"""
function calculate_logprob2(::TrajectoryNUTS, is_doubling, ω₁, ω₂, ω)
    return biased_progressive_logprob2(is_doubling, ω₁, ω₂, ω)
end

"""
$(SIGNATURES)
Combine two proposals `ζ₁, ζ₂` on `trajectory`, with log probability `logprob2` for
selecting `ζ₂`.
 `ζ₁` is before `ζ₂` iff `is_forward`.

# Examples
```julia
```

"""
function combine_proposals(
    _rng::Random.AbstractRNG,
    trajectory::TrajectoryNUTS,
    z₁::PhasePoint,
    z₂::PhasePoint,
    logprob2::T,
    is_forward,
) where {T<:Real}
    return rand_bool_logprob(_rng, logprob2) ? z₂ : z₁
end

############################################################################################
# statistics for visited nodes
struct AcceptanceStatistic{F<:AbstractFloat}
    "
    Logarithm of the sum of metropolis acceptances probabilities over the whole trajectory
    (including invalid parts).
    "
    log_sum_α::F
    "Total number of leapfrog steps."
    steps::Int64
end

function combine_acceptance_statistics(A::AcceptanceStatistic, B::AcceptanceStatistic)
    return AcceptanceStatistic(
        BaytesCore.logaddexp(A.log_sum_α, B.log_sum_α), A.steps + B.steps
    )
end

"""
$(SIGNATURES)
Acceptance statistic for a leaf. The initial leaf is considered not to be visited.

# Examples
```julia
```

"""
function leaf_acceptance_statistic(Δ, is_initial)
    return if is_initial
        AcceptanceStatistic(oftype(Δ, -Inf), 0)
    else
        AcceptanceStatistic(min(Δ, 0), 1)
    end
end

"""
$(SIGNATURES)
Return the acceptance rate (a `Real` betwen `0` and `1`).

# Examples
```julia
```

"""
acceptance_rate(A::AcceptanceStatistic) = min(exp(A.log_sum_α) / A.steps, 1)

"""
$(SIGNATURES)
Combine visited node statistics for adjacent trees trajectory. Implementation should be
invariant to the ordering of `v₁` and `v₂` (ie the operation is commutative).

# Examples
```julia
```

"""
combine_visited_statistics(::TrajectoryNUTS, v, w) = combine_acceptance_statistics(v, w)

############################################################################################
# abstract trajectory interface
"""
$(TYPEDEF)
Information about an invalid (sub)tree, using positions relative to the starting node.
1. When `left < right`, this tree was *turning*.
2. When `left == right`, this is a *divergent* node.
3. `left == 1 && right == 0` is used as a sentinel value for reaching maximum depth without
encountering any invalid trees (see [`REACHED_MAX_DEPTH`](@ref). All other `left > right`
values are disallowed.

# Fields
$(TYPEDFIELDS)
"""
struct InvalidTree
    left::Int64
    right::Int64
end
InvalidTree(i::Integer) = InvalidTree(i, i)
is_divergent(invalid_tree::InvalidTree) = invalid_tree.left == invalid_tree.right

############################################################################################
# Directions

"""
$(TYPEDEF)
Internal type implementing random directions.

# Fields
$(TYPEDFIELDS)
"""
struct Directions
    flags::UInt32
end

Base.rand(_rng::Random.AbstractRNG, ::Type{Directions}) = Directions(rand(_rng, UInt32))

"""
$(SIGNATURES)
Return the next direction flag and the new state of directions. Results are undefined for
more than [`MAX_DIRECTIONS_DEPTH`](@ref) updates.

# Examples
```julia
```

"""
function next_direction(directions::Directions)
    @unpack flags = directions
    return Bool(flags & 0x01), Directions(flags >>> 1)
end

"Sentinel value for reaching maximum depth."
const REACHED_MAX_DEPTH = InvalidTree(1, 0)

################################################################################
# turn analysis

"""
$(TYPEDEF)
Statistics for the identification of turning points. See Betancourt (2017, appendix), and
subsequent discussion of improvements at
<https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/>.
Momenta `p₋` and `p₊` are kept so that they can be added to `ρ` when combining turn
statistics.
Turn detection is always done by [`combine_turn_statistics`](@ref), which returns `nothing`
in case of turning. A `GeneralizedTurnStatistic` should always correspond to a trajectory
that is *not* turning (or a leaf node, where the concept does not apply).

# Fields
$(TYPEDFIELDS)
"""
struct GeneralizedTurnStatistic{T}
    "momentum at the left edge of the trajectory"
    ρ₋::T
    "p♯ at the left edge of the trajectory"
    ρ♯₋::T
    "momentum at the right edge of the trajectory"
    ρ₊::T
    "p♯ at the right edge of the trajectory"
    ρ♯₊::T
    "sum of momenta along trajectory"
    ρₛ::T
end

function leaf_turn_statistic(::Val{:generalized}, H::Hamiltonian, phasepoint::PhasePoint)
    @unpack ρ = phasepoint
    ρ♯ = calculate_ρ♯(H, phasepoint)
    return GeneralizedTurnStatistic(ρ, ρ♯, ρ, ρ♯, ρ)
end

"""
Internal test for turning. See Betancourt (2017, appendix).
"""
_is_turning(ρ♯₋, ρ♯₊, ρ) = dot(ρ♯₋, ρ) < 0 || dot(ρ♯₊, ρ) < 0

"""
$(SIGNATURES)
Combine turn statistics on trajectory. Implementation can assume that the trees that
correspond to the turn statistics have the same ordering.
When
```julia
τ = combine_turn_statistics(trajectory, τ₁, τ₂)
is_turning(trajectory, τ)
```
the combined turn statistic `τ` is guaranteed not to escape the caller, so it can eg change
type.

# Examples
```julia
```

"""
function combine_turn_statistics(
    trajectory::TrajectoryNUTS, x::GeneralizedTurnStatistic, y::GeneralizedTurnStatistic
)
    _is_turning(x.ρ♯₋, y.ρ♯₋, x.ρₛ + y.ρ₋) && return nothing
    _is_turning(x.ρ♯₊, y.ρ♯₊, x.ρ₊ + y.ρₛ) && return nothing
    ρₛᵤₘ = x.ρₛ + y.ρₛ
    #!NOTE: Most left and Right Momenta
    _is_turning(x.ρ♯₋, y.ρ♯₊, ρₛᵤₘ) && return nothing
    return GeneralizedTurnStatistic(x.ρ₋, x.ρ♯₋, y.ρ₊, y.ρ♯₊, ρₛᵤₘ)
end

"""
Internal test for turning. See Betancourt (2017, appendix).
"""
is_turning(::TrajectoryNUTS, ::GeneralizedTurnStatistic) = false
is_turning(::TrajectoryNUTS, ::Nothing) = true

############################################################################################
# utilities

"""
$(SIGNATURES)
Combine turn statistics with the given direction. When `is_forward`, `τ₁` is before `τ₂`,
otherwise after. Internal helper function.

# Examples
```julia
```

"""
@inline function combine_turn_statistics_in_direction(
    trajectory::TrajectoryNUTS,
    τ₁::GeneralizedTurnStatistic,
    τ₂::GeneralizedTurnStatistic,
    is_forward::Bool,
)
    if is_forward
        combine_turn_statistics(trajectory, τ₁, τ₂)
    else
        combine_turn_statistics(trajectory, τ₂, τ₁)
    end
end

function combine_proposals_and_logweights(
    _rng::Random.AbstractRNG,
    trajectory::TrajectoryNUTS,
    ζ₁::PhasePoint,
    ζ₂::PhasePoint,
    ω₁::R,
    ω₂::R,
    is_forward::Bool,
    is_doubling::Bool,
) where {R<:Real}
    ω = BaytesCore.logaddexp(ω₁, ω₂)
    logprob2 = calculate_logprob2(trajectory, is_doubling, ω₁, ω₂, ω)
    ζ = combine_proposals(_rng, trajectory, ζ₁, ζ₂, logprob2, is_forward)
    return ζ, ω
end

############################################################################################
# leafs

"""
$(SIGNATURES)
Information for a tree made of a single node. When `is_initial == true`, this is the first
node.
The first value is either
1. `nothing` for a divergent node,
2. a tuple containing the proposal `ζ`, the log weight (probability) of the node `ω`, the
turn statistics `τ` (never tested with `is_turning` for leaf nodes).
The second value is the visited node information.

# Examples
```julia
```

"""
function leaf(trajectory::TrajectoryNUTS, phasepoint::PhasePoint, is_initial)
    @unpack H, ℓH₀, min_Δ, turn_statistic_configuration = trajectory
    Δ = is_initial ? zero(ℓH₀) : ℓdensity(trajectory.H, phasepoint) - ℓH₀
    isdiv = Δ < min_Δ
    v = leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        nothing, v
    else
        τ = leaf_turn_statistic(turn_statistic_configuration, H, phasepoint)
        (phasepoint, Δ, τ), v
    end
end

############################################################################################
"""
$(SIGNATURES)
Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.
`is_forward` specifies the direction, `rng` is used for random numbers in
[`combine_proposals`](@ref). `i` is an integer position relative to the initial node (`0`).
The *first value* is either
1. an `InvalidTree`, indicating the first divergent node or turning subtree that was
encounteted and invalidated this tree.
2. a tuple of `(ζ, ω, τ, z′, i′), with
    - `ζ`: the proposal from the tree.
    - `ω`: the log weight of the subtree that corresponds to the proposal
    - `τ`: turn statistics
    - `z′`: the last node of the tree
    - `i′`: the position of the last node relative to the initial node.
The *second value* is always the visited node statistic.

# Examples
```julia
```

"""
function adjacent_tree(
    _rng::Random.AbstractRNG,
    trajectory::TrajectoryNUTS,
    z::PhasePoint,
    i::Int64,
    depth::Int64,
    is_forward::Bool,
)
    i′ = i + (is_forward ? 1 : -1)
    if depth == 0
        z′ = move(trajectory, z, is_forward)
        ζωτ, v = leaf(trajectory, z′, false)
        if ζωτ ≡ nothing
            InvalidTree(i′), v
        else
            (ζωτ..., z′, i′), v
        end
    else
        ## “left” tree
        t₋, v₋ = adjacent_tree(_rng, trajectory, z, i, depth - 1, is_forward)
        t₋ isa InvalidTree && return t₋, v₋
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        ## “right” tree — visited information from left is kept even if invalid
        t₊, v₊ = adjacent_tree(_rng, trajectory, z₋, i₋, depth - 1, is_forward)
        v = combine_visited_statistics(trajectory, v₋, v₊)
        t₊ isa InvalidTree && return t₊, v
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        ## turning invalidates
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        is_turning(trajectory, τ) && return InvalidTree(i′, i₊), v

        ## valid subtree, combine proposals
        ζ, ω = combine_proposals_and_logweights(
            _rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false
        )
        (ζ, ω, τ, z₊, i₊), v
    end
end

"""
$(SIGNATURES)
Sample a `trajectory` starting at `z`, up to `max_depth`. `directions` determines the tree
expansion directions.
Return the following values
- `ζ`: proposal from the tree
- `v`: visited node statistics
- `termination`: an `InvalidTree` (this includes the last doubling step turning, which is
  technically a valid tree) or `REACHED_MAX_DEPTH` when all subtrees were valid and no
  turning happens.
- `depth`: the depth of the tree that was sampled from. Doubling steps that lead to an
  invalid adjacent tree do not contribute to `depth`.

# Examples
```julia
```

"""
function sample_trajectory(
    _rng::Random.AbstractRNG,
    trajectory::TrajectoryNUTS,
    z::PhasePoint,
    directions::Directions,
)
    @unpack max_depth = trajectory
    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    (ζ, ω, τ), v = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = REACHED_MAX_DEPTH
    i₋ = i₊ = 0
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        t′, v′ = adjacent_tree(
            _rng, trajectory, is_forward ? z₊ : z₋, is_forward ? i₊ : i₋, depth, is_forward
        )
        v = combine_visited_statistics(trajectory, v, v′)

        ## invalid adjacent tree: stop
        t′ isa InvalidTree && (termination = t′; break)

        ## extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′

        ## update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        ## tree has doubled successfully
        ζ, ω = combine_proposals_and_logweights(
            _rng, trajectory, ζ, ζ′, ω, ω′, is_forward, true
        )
        depth += 1

        ## when the combined tree is turning, stop
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_turning(trajectory, τ) && (termination = InvalidTree(i₋, i₊);
        break)
    end
    return ζ, v, termination, depth
end

############################################################################################
#export
export TrajectoryNUTS,
    biased_progressive_logprob2,
    rand_bool_logprob,
    calculate_logprob2,
    combine_proposals,
    AcceptanceStatistic,
    combine_acceptance_statistics,
    leaf_acceptance_statistic,
    acceptance_rate,
    combine_visited_statistics,
    InvalidTree,
    is_divergent,
    Directions,
    next_direction,
    REACHED_MAX_DEPTH,
    GeneralizedTurnStatistic,
    leaf_turn_statistic,
    _is_turning,
    combine_turn_statistics,
    combine_turn_statistics_in_direction,
    combine_proposals_and_logweights,
    leaf,
    adjacent_tree,
    sample_trajectory
