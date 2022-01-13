# BaytesMCMC

<!---
![logo](docs/src/assets/logo.svg)

[![Documentation, Stable](xxx)](xxx)
[![Documentation, Latest](xxx)](xxx)
[![CI](xxx)](xxx)
[![Codecov](xxx)](xxx)
[![arXiv article](xxx)](xxx)
-->

BaytesMCMC.jl is a library to perform MCMC proposal steps on `ModelWrapper` structs, see [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl).

<!---
[BaytesMCMC.jl](xxx)
[BaytesFilters.jl](xxx)
[BaytesPMCMC.jl](xxx)
[BaytesSMC.jl](xxx)
[Baytes.jl](xxx)
-->

## First steps

Let us use the model initially defined in the [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl) introduction:
```julia
using ModelWrappers, BaytesMCMC
using Distributions, Random, UnPack
_rng = Random.GLOBAL_RNG
#Create Model and data
myparameter = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(myparameter)
data = randn(1000)
#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
```

We can assign an MCMC Kernel to this model by simply calling
```julia
mcmc_nuts = MCMC(NUTS, myobjective)
mcmc_hmc = MCMC(HMC, myobjective)
mcmc_mala = MCMC(MALA, myobjective)
mcmc_metropolis = MCMC(Metropolis, myobjective)
```

There are two standard ways to make proposal steps in BaytesMCMC.

1. The first step is the standard MCMC proposal step:
```julia
_val, _diagnostics = propose(_rng, mcmc_nuts, myobjective)
```
This will return proposed model parameter (if accepted), along with MCMC summary diagnostics.

2. The second way can be used if the MCMC kernel is used alongside other algorithms, or if the data changes between proposal steps.
In this case, the MCMC kernel is updated with the paramter given in `mymodel` and the new `data` before a proposal step is performed. This
will take up more time then the first way, but is more flexible and can be used in scenarios where, for instance, you want to use MCMC
in combination with other inference algorithm.
```julia
_val, _diagnostics = propose!(_rng, mcmc_nuts, mymodel, data)
```
This will update `mymodel` with the proposed parameter (if accepted), and return `mymodel` parameter along with MCMC summary diagnostics.

## Customization

All MCMC kernels are initialized with sane default tuning parameter, but each field in the config.jl file of each sampler in src/Kernels can be fully customized. For instance, the following settings initializes a HMC kernel with fixed stepsize and a dense mass matrix adaption. Moreover, the ReverseDiff package is used for obtaining derivative information.
```julia
mcmcdefault = MCMCDefault(;
  config_kw = (; ϵ = 1.0, stepsizeadaption = UpdateFalse(), steps = 10, metric = MDense()), GradientBackend = :ReverseDiff,
)

mcmc_customized = MCMC(HMC, myobjective; default = mcmcdefault)
_val, _diagnostics = propose(_rng, mcmc_customized, myobjective)
_val, _diagnostics = propose!(_rng, mcmc_customized, mymodel, data)
```
There are many customization options for each kernel, which can be seen in the corresponding config.jl files.

## Generated data and prediction

You can return generated quantities and predictions of your model for each proposal step. Just like with the objective functor, you need to add methods for your model for the corresponding functions:
```julia
function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack μ, σ = objective.model.val
    return rand(_rng, Normal(μ, σ))
end
function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack model, data = objective
    # Return some interesting stuff
    return model.val.μ + randn(_rng)
end

predict(_rng, myobjective)
generate(_rng, myobjective)

mcmc_nuts = MCMC(NUTS, myobjective; default = MCMCDefault(generated = true))
_val, _diagnostics = propose(_rng, mcmc_nuts, myobjective)
_diagnostics.prediction
_diagnostics.generated
```

## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->

## License Notice

Note that this package heavily uses and adapts code from the DynamicHMC.jl package (v3.1.0) licensed under MIT License. In particular,
the files stepsize.jl and folders src/Kernels/NUTS have been heavily adapted and used.

> Copyright (c) 2020: Tamas K. Papp.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
