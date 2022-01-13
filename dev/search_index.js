var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be properly done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BaytesMCMC","category":"page"},{"location":"#BaytesMCMC","page":"Home","title":"BaytesMCMC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BaytesMCMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BaytesMCMC]","category":"page"},{"location":"#BaytesMCMC.AcceptStatistic","page":"Home","title":"BaytesMCMC.AcceptStatistic","text":"struct AcceptStatistic{T<:AbstractFloat}\n\nStores in diagnostics.\n\nFields\n\nrate::AbstractFloat\nAcceptance rate\naccepted::Bool\nStep accepted or rejected\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.Adaptionˢˡᵒʷ","page":"Home","title":"BaytesMCMC.Adaptionˢˡᵒʷ","text":"Explore typical set and tune MCMC parameter(initial bias rapidly vanishes and the estimators become much more accurate.)\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.Adaptionᶠᵃˢᵗ","page":"Home","title":"BaytesMCMC.Adaptionᶠᵃˢᵗ","text":"Tune final stepsize for fixed Covariance Matrix.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.DualAverage","page":"Home","title":"BaytesMCMC.DualAverage","text":"struct DualAverage{T<:AbstractFloat}\n\nContains DualAverage tuning information and runtime parameter.\n\nFields\n\nadaption::DualAverageParameter\nμ::AbstractFloat\nUpwards bias for target acceptance rate - proposals are biased upwards to stay away from 0.\nt::Int64\nTime update, starts with 0\nH̄::AbstractFloat\nAverage part of first equation in Hoffman(2014), p 1607, (6)\nlogϵ::AbstractFloat\nLog step\nlogϵ̄::AbstractFloat\nAVERAGED log step\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.DualAverageParameter","page":"Home","title":"BaytesMCMC.DualAverageParameter","text":"struct DualAverageParameter{T<:AbstractFloat}\n\nContains information for default Dual Averaging algorithm.\n\nFields\n\nδ::AbstractFloat\nTarget acceptance rate\nγ::AbstractFloat\nRegularization scale\nκ::AbstractFloat\nRelaxation exponent - for Average log step size\nt₀::Int64\nOffset\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.EuclideanKineticEnergy","page":"Home","title":"BaytesMCMC.EuclideanKineticEnergy","text":"Gaussian Kinetic Energy, independent of position parameter.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.Exploration","page":"Home","title":"BaytesMCMC.Exploration","text":"Sample parameter with tuned MCMC (gradually reducing the precision error of the MCMC estimators towards zero.)\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.GaussianKineticEnergy","page":"Home","title":"BaytesMCMC.GaussianKineticEnergy","text":"mutable struct GaussianKineticEnergy{T<:(AbstractMatrix), S<:(AbstractMatrix)} <: EuclideanKineticEnergy\n\nGaussian kinetic energy, which is independent of q.\n\nFields\n\nΣ::AbstractMatrix\nInverse Mass Matrix Σ ~ Posterior Covariance Matrix\nMᶜʰᵒˡ::AbstractMatrix\nCholesky decomposition of Mass matrix M, s.t. Mᶜʰᵒˡ*Mᶜʰᵒˡ'=M. Used to generate random draws\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.GaussianKineticEnergy-Tuple{AbstractMatrix}","page":"Home","title":"BaytesMCMC.GaussianKineticEnergy","text":"Gaussian kinetic energy with the given inverse covariance matrix Σ.\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.InitialStepsizeSearch","page":"Home","title":"BaytesMCMC.InitialStepsizeSearch","text":"struct InitialStepsizeSearch{T<:AbstractFloat}\n\nInitial Step Size Search factors.\n\nFields\n\na_min::AbstractFloat\nLowest local acceptance rate.\na_max::AbstractFloat\nHighest local acceptance rate.\nϵ₀::AbstractFloat\nInitial stepsize.\nC::AbstractFloat\nScale factor for initial bracketing, > 1.\nmaxiter_crossing::Int64\nMaximum number of iterations for initial bracketing.\nmaxiter_bisect::Int64\nMaximum number of iterations for bisection.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.KineticEnergy","page":"Home","title":"BaytesMCMC.KineticEnergy","text":"Kinetic Energy in HMC setting.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MCMC","page":"Home","title":"BaytesMCMC.MCMC","text":"struct MCMC{M<:MCMCKernel, N<:MCMCTune} <: BaytesCore.AbstractAlgorithm\n\nStores information for proposal step.\n\nFields\n\nkernel::MCMCKernel\nMCMC sampler\ntune::MCMCTune\nTuning configuration for kernel.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MCMCConstructor","page":"Home","title":"BaytesMCMC.MCMCConstructor","text":"Callable struct to make initializing MCMC sampler easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MCMCDefault","page":"Home","title":"BaytesMCMC.MCMCDefault","text":"struct MCMCDefault{K<:NamedTuple, S}\n\nDefault arguments for MCMC constructor.\n\nFields\n\nconfig_kw::NamedTuple\nIndividual keyword arguments for tuning different MCMC engines.\nGradientBackend::Any\nGradient backend used in MCMC step. Not used if Metropolis sampler is chosen.\nTunedModel::Bool\nBoolean if initial parameter are fixed or resampled.\ngenerated::Bool\nBoolean if generate(_rng, objective) for corresponding model is stored in MCMC Diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MCMCDiagnostics","page":"Home","title":"BaytesMCMC.MCMCDiagnostics","text":"struct MCMCDiagnostics{R<:AbstractFloat, E<:MCMCKernelDiagnostics, T, G} <: BaytesCore.AbstractDiagnostics\n\nMCMC Diagnostics container.\n\nFields\n\nℓθᵤ::AbstractFloat\nEvaluation of target function at current iteration.\ntemperature::AbstractFloat\nTemperature for log posterior evaluation\ndivergence::Bool\nBoolean if diverged.\naccept::AcceptStatistic\nAcceptance Rate of current step.\nsampler::MCMCKernelDiagnostics\nSampler specific diagnostics.\nprediction::Any\nPredicted sample of model.\ngenerated::Any\nGenerated quantities specified for objective\niter::Int64\nCurrent iteration number.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MCMCTune","page":"Home","title":"BaytesMCMC.MCMCTune","text":"struct MCMCTune{A<:UpdateBool, B<:UpdateBool, F<:AbstractFloat, T<:ModelWrappers.Tagged, E<:Tuple, P<:Proposal} <: BaytesCore.AbstractTune\n\nMCMC Tuning container.\n\nFields\n\ntagged::ModelWrappers.Tagged\nTagged Parameter.\nphase::PhaseTune\nCurrent Phase in MCMC Cycle\nstepsize::StepSizeTune\nStepsize container\nproposal::Proposal\nInformation for posterior covariance estimate\ntempering::BaytesCore.TemperingTune\nInformation about tempering target function.\ngenerated::Bool\nBoolean if generated quantities should be generated while sampling\niter::BaytesCore.Iterator\nCurrent iteration number\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MatrixMetric","page":"Home","title":"BaytesMCMC.MatrixMetric","text":"abstract type MatrixMetric\n\nChoice for Posterior Covariance Matrix adaption.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.MatrixTune","page":"Home","title":"BaytesMCMC.MatrixTune","text":"struct MatrixTune{A<:MatrixMetric}\n\nMass and Covariance Matrix specification for MCMC sampler, relevant for Euclidean Metric.\n\nFields\n\nmetric::MatrixMetric\nDense, Diagonal or Unit Mass Matrix.\nshrinkage::Float64\nShrinkage parameter for Covariance estimation.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.PhaseTune","page":"Home","title":"BaytesMCMC.PhaseTune","text":"struct PhaseTune{T<:Tuple}\n\nInformation about current SamplingPhase.\n\nFields\n\nupdate::BaytesCore.Updater\nBoolean if current iteration needs update.\niter::BaytesCore.Iterator\nCounts current iteration in phase.\ncounter::BaytesCore.Iterator\nMCMC Phases ~ counter = current cyle in slices/name/iterations.\nslices::Vector{Int64}\nVector of MCMC iterations at each phase.\nname::Tuple\nName of Sampling phases.\niterations::Vector{Int64}\nCounts total iterations.\nwindow::Vector{Int64}\nCounts cycles in adaption phases Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(),     i.e., 1-5-1 means 1 window init, 5 adapt, 1 exploration.\nbuffer::Vector{Int64}\nCounts iteration in each cycle for adaption phases Warmup(), Adaptionˢˡᵒʷ(), Adaptionᶠᵃˢᵗ(),     i.e., 50-25-50 means 1 time 50, 5 times 25*i, i in 1:5, 1 time 50.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.Proposal","page":"Home","title":"BaytesMCMC.Proposal","text":"mutable struct Proposal{A<:UpdateBool, T<:AbstractFloat, P<:(AbstractMatrix), C<:(AbstractMatrix), M<:MatrixTune}\n\nProposal distribution container.\n\nFields\n\nadaption::UpdateBool\nCheck if adaption true in current iteration\nchain::Matrix{T} where T<:AbstractFloat\nθᵤ samples in current MCMC Phase, used for Σ estimation\nΣ::AbstractMatrix\nPosterior Covariance estimate\nΣ⁻¹ᶜʰᵒˡ::AbstractMatrix\nCholesky decomposition of Inverse Posterior Covariance matrix Σ, s.t. Σ⁻¹ᶜʰᵒˡ*Σ⁻¹ᶜʰᵒˡ'=Σ⁻¹. Used to generate random draws in HMC/NUTS sampler.\nmatrixtune::MatrixTune\nTuning parameter for Σ estimation\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.SamplingPhase","page":"Home","title":"BaytesMCMC.SamplingPhase","text":"abstract type SamplingPhase\n\nSampling Container that determines tuning updates.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.StepSizeTune","page":"Home","title":"BaytesMCMC.StepSizeTune","text":"mutable struct StepSizeTune{A<:UpdateBool, F<:AbstractFloat}\n\nTuning stepsize parameter for MCMC algorithm.\n\nFields\n\nadaption::UpdateBool\nIf true, stepsize will be adapted.\ndualaverage::DualAverage\nDualaverage struct\nϵ::AbstractFloat\nCurrent stepsize\n\n\n\n\n\n","category":"type"},{"location":"#BaytesMCMC.Warmup","page":"Home","title":"BaytesMCMC.Warmup","text":"Find/converges to the typical set (estimators suffer from initial but ultimately transient biases)\n\n\n\n\n\n","category":"type"},{"location":"#BaytesCore.adapt!-Union{Tuple{P}, Tuple{P, MDense}} where P<:Proposal","page":"Home","title":"BaytesCore.adapt!","text":"adapt!(proposal, metric)\nadapt!(proposal)\n\n\nCalculate regularized Covariance Matrix and Cholesky decomposition.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, MCMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"Infer type of predictions of MCMC sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, MCMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, mcmc, model, data)\n\n\nInfer MCMC diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.init-Union{Tuple{T}, Tuple{Type{DualAverage}, Any, T}} where T<:AbstractFloat","page":"Home","title":"BaytesCore.init","text":"init(, δ, ϵ; multiplier)\n\n\nInitialize new DualAverage struct.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{U}, Tuple{D}, Tuple{Random.AbstractRNG, MCMC, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, MCMC, ModelWrappers.ModelWrapper, D, U}} where {D, U<:UpdateBool}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, mcmc, model, data)\npropose!(_rng, mcmc, model, data, update)\n\n\nInplace version of propose.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose-Tuple{Random.AbstractRNG, MCMC, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propose","text":"propose(_rng, mcmc, objective)\n\n\nPropose new parameter with mcmc sampler. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.results-Union{Tuple{M}, Tuple{T}, Tuple{AbstractVector{M}, MCMC, Integer, Vector{T}}} where {T<:Real, M<:MCMCDiagnostics}","page":"Home","title":"BaytesCore.results","text":"results(diagnosticsᵛ, mcmc, Ndigits, quantiles)\n\n\nPrint result for a single trace.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Tuple{PhaseTune, Int64}","page":"Home","title":"BaytesCore.update!","text":"update!(phase, iter)\n\n\nUpdate current MCMC Phase.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{N}, Tuple{BaytesCore.TemperingTune, UpdateTrue, N, Integer}} where N<:Union{Adaptionˢˡᵒʷ, Warmup}","page":"Home","title":"BaytesCore.update!","text":"update!(tempering, adaption, phasename, iter)\n\n\nUpdate Tempering with new temperature.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{P}, Tuple{P, UpdateFalse, Any, Any, Any}} where P<:Proposal","page":"Home","title":"BaytesCore.update!","text":"update!(proposal, proposalupdate, θᵤ, phasename, phase)\n\n\nUpdate Proposal according to current tuning phase.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{T}, Tuple{N}, Tuple{P}, Tuple{P, Val{true}, N, PhaseTune, AbstractVector{T}}} where {P<:Proposal, N<:Union{Adaptionˢˡᵒʷ, Adaptionᶠᵃˢᵗ, Warmup}, T<:Real}","page":"Home","title":"BaytesCore.update!","text":"update!(proposal, phaseupdate, phasename, phase, θᵤ)\n\n\nUpdate Proposal with new parameter θᵤ.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{T}, Tuple{N}, Tuple{StepSizeTune, UpdateTrue, T, N, Val{false}}} where {N<:Union{Adaptionˢˡᵒʷ, Adaptionᶠᵃˢᵗ, Warmup}, T<:AbstractFloat}","page":"Home","title":"BaytesCore.update!","text":"update!(stepsize, stepsizeadaption, α, samplingphase, iterationupdate)\n\n\nUpdate stepsize based on acceptance ratio α.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{T}, Tuple{S}, Tuple{MCMCTune, S, T}} where {S<:ModelWrappers.ℓObjectiveResult, T<:Real}","page":"Home","title":"BaytesCore.update!","text":"update!(tune, result, acceptrate)\n\n\nUpdate MCMC tuning fields at current iteration.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update-Union{Tuple{T}, Tuple{DualAverage, T}} where T<:AbstractFloat","page":"Home","title":"BaytesCore.update","text":"update(dualaverage, acceptratio)\n\n\nUpdate DualAverage struct.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC._infer_generated-Union{Tuple{D}, Tuple{Random.AbstractRNG, MCMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesMCMC._infer_generated","text":"_infer_generated(_rng, mcmc, model, data)\n\n\nInfer type of generated quantities of MCMC sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.bisect_stepsize","page":"Home","title":"BaytesMCMC.bisect_stepsize","text":"bisect_stepsize(parameters, A, ϵ₀, ϵ₁)\nbisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀)\nbisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀, Aϵ₁)\n\n\nReturn the desired stepsize ϵ by bisection.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#BaytesMCMC.calculate_ρ♯","page":"Home","title":"BaytesMCMC.calculate_ρ♯","text":"Return p♯ = M⁻¹⋅p, used for turn diagnostics.\n\n\n\n\n\n","category":"function"},{"location":"#BaytesMCMC.chain!-Union{Tuple{N}, Tuple{P}, Tuple{P, N, Integer}} where {P<:Proposal, N<:Union{Adaptionˢˡᵒʷ, Warmup}}","page":"Home","title":"BaytesMCMC.chain!","text":"chain!(proposal, phasename, Niterations)\n\n\nAssign new chain buffer with dedicated size.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.find_crossing_stepsize","page":"Home","title":"BaytesMCMC.find_crossing_stepsize","text":"find_crossing_stepsize(parameters, A, ϵ₀)\nfind_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀)\n\n\nFind the stepsize for which the local acceptance rate A(ϵ) crosses a.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#BaytesMCMC.find_initial_stepsize-Tuple{BaytesMCMC.InitialStepsizeSearch, Any}","page":"Home","title":"BaytesMCMC.find_initial_stepsize","text":"find_initial_stepsize(parameters, A)\n\n\nFind an initial stepsize that matches the conditions of parameters (see InitialStepsizeSearch). A is the local acceptance ratio (unbounded).\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.get_Σ-Union{Tuple{T}, Tuple{D}, Tuple{D, T}} where {D<:Union{LinearAlgebra.Diagonal, LinearAlgebra.Symmetric}, T<:Real}","page":"Home","title":"BaytesMCMC.get_Σ","text":"get_Σ(Σ, shrinkage)\n\n\nCalculate regularized Covariance Matrix.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.print_divergences-Union{Tuple{M}, Tuple{AbstractVector{M}, PhaseTune}} where M<:MCMCDiagnostics","page":"Home","title":"BaytesMCMC.print_divergences","text":"print_divergences(diagnosticsᵛ, phase)\n\n\nPrint all divergences from a vector of MCMC Diagnostics.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesMCMC.printtune","page":"Home","title":"BaytesMCMC.printtune","text":"printtune(tune)\nprinttune(tune, diagparam)\n\n\nSome basic tuning output figures. Useful for debugging.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#BaytesMCMC.rand_ρ","page":"Home","title":"BaytesMCMC.rand_ρ","text":"Generate a random momentum from a kinetic energy at position ρ.\n\n\n\n\n\n","category":"function"},{"location":"#BaytesMCMC.∇K","page":"Home","title":"BaytesMCMC.∇K","text":"Calculate the gradient of the logarithm of kinetic energy in momentum ρ.\n\n\n\n\n\n","category":"function"}]
}
