############################################################################################
# Constants
"RNG for sampling based solutions"
const _rng = Random.Xoshiro(123)  # shorthand
Random.seed!(_rng, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3

############################################################################################
# Kernel and AD backends
kernels = [NUTS, HMC, MALA, Metropolis, Custom]
gradientkernels = [NUTS, HMC, MALA]

backends = [:ForwardDiff, :ReverseDiff, :ReverseDiffUntaped, :Zygote]
massmatrices = [MDense(), MDiagonal(), MUnit()]
generating = [UpdateFalse(), UpdateTrue()]

############################################################################################
# Initiate Base Model to check sampler

######################################## Model 1
data_uv = randn(_rng, 1000)

struct MyBaseModel <: ModelName end
myparameter = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(MyBaseModel(), myparameter)

#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data_uv, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{MyBaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
myobjective(myobjective.model.val)

function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
    return μ[1]
end

function ModelWrappers.generate(_rng::Random.AbstractRNG, algorithm::MCMC, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
    return μ[1] + 100000
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
	return rand(_rng, Normal(μ, σ))
end
