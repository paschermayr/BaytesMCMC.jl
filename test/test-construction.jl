############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter, (;), FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter, (;), FlattenDefault(; output = Float32)), data_uv)
]
#=
iter = 2
backend = backends[end]
matrixadaption = massmatrices[1]
kernel = kernels[1]
generated = generating[2]
=#
############################################################################################
## Make model
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.reconstruct.default.output
    @testset "Kernel construction and propagation, all models" begin
        ## MCMC AD backends
        for backend in backends
            ## Mass matrix adaption settings
            for matrixadaption in massmatrices
                for generated in generating
                    ## Define MCMC default tuning parameter
                    mcmcdefault = MCMCDefault(;
                        stepsize = ConfigStepsize(; stepsizeadaption=UpdateFalse()),
                        proposal = ConfigProposal(; metric = matrixadaption),
                        GradientBackend = backend,
                        generated = generated)
                    ## MCMC kernels
                    for kernel in kernels
                        ## Check if Default options work
                        MCMC(_rng, kernel, _obj, mcmcdefault)
                        MCMC(_rng, kernel, _obj, mcmcdefault, SampleDefault())
                        MCMC(kernel, keys(_obj.tagged.parameter))
                        ## Check if constructors work
                        MCMCConstructor(kernel, keys(_obj.tagged.parameter)[begin], mcmcdefault)
                        constructor = MCMCConstructor(kernel, keys(_obj.tagged.parameter), mcmcdefault)
                        constructor(_rng, _obj.model, _obj.data, BaytesCore.ProposalTune(_flattentype(1.0)), SampleDefault())
                        ## Initialize kernel and check if it can be run
                        mcmckernel = MCMC(
                            _rng,
                            kernel,
                            _obj,
                            mcmcdefault
                        )
                        _val1, _diag1 = propose(_rng, mcmckernel, _obj)
                        _val2, _diag2 = propose!(_rng, mcmckernel, _obj.model, _obj.data)
                        ## Test if Float types for proposal calculation all have same type after proposal step
                        # !NOTE kernel.result structs are already checked in ModelWrappers
                        @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                        @test eltype(mcmckernel.tune.proposal.Σ) ==
                            eltype(mcmckernel.tune.proposal.Σ⁻¹ᶜʰᵒˡ) ==
                            eltype(mcmckernel.tune.proposal.chain) ==  _flattentype

                        ## Postprocessing
                        @test _diag1 isa infer(_rng, AbstractDiagnostics, mcmckernel, _obj.model, _obj.data)
                        @test _diag2 isa infer(_rng, AbstractDiagnostics, mcmckernel, _obj.model, _obj.data)
                        @test _diag1.base.prediction isa infer(_rng, mcmckernel, _obj.model, _obj.data)
                        generated_model, generated_algorithm = BaytesMCMC.infer_generated(_rng, mcmckernel, _obj.model, _obj.data)
                        @test _diag1.generated isa generated_model
                        @test _diag1.generated_algorithm isa generated_algorithm
                        results([_diag1, _diag2], mcmckernel, 2, [.1, .2, .5, .8, .9])
                        divs = BaytesMCMC.print_divergences([_diag1, _diag2], mcmckernel.tune.phase);
                        @test length(divs[1]) == length(divs[2])
                        BaytesMCMC.result!(mcmckernel, BaytesMCMC.get_result(mcmckernel))
                        generate_showvalues(_diag1)()

                        ## Tuning settings
                        #!NOTE: We do not need to check this for all backends as separate propose calls already evaluated and tune updates are independent of gradients
                        if backend == backends[1]
                            for iter in Base.OneTo(mcmckernel.tune.phase.slices[end])
                                propose(_rng, mcmckernel, _obj)
                            end
                        end

                        ## Check if MCMC also works with more/less data
                        propose!(_rng, mcmckernel, _obj.model, randn(_rng, length(data_uv)+10))
                        propose!(_rng, mcmckernel, _obj.model, randn(_rng, length(data_uv)-10))

                    end
                end
            end
        end
    end
end

############################################################################################
#!NOTE: Test Enzyme framework separately for now
struct EnzymeBM <: ModelName end
_paramEnzyme = (μ=Param(Distributions.Normal(), 1.,), σ=Param(Distributions.Exponential(), 2.,))
_a = 1.
_b = collect(1.:10.)
_args = (a = copy(_a), b = copy(_b) )
dat = randn(100)
dat[1:5] = collect(1:5) .+ 0.0
dat

function (objective::Objective{<:ModelWrapper{EnzymeBM}})(θ::NamedTuple, arg::A = objective.model.arg, data::D = objective.data) where {A, D}
    μ = θ.μ + arg.a + mean(arg.b)
    #=
    #!NOTE: Up until at least Enzyme 10.16, Enzyme mutates model.arg and objective.data even if specified as constant
    lp =
        Distributions.logpdf(Distributions.Normal(), μ) +
        Distributions.logpdf(Distributions.Exponential(), θ.σ)
    ll = sum( logpdf(Normal(μ, θ.σ), dat) for dat in data )
    return ll #+ lp
    =#
    ll = sum( logpdf(Normal(μ, θ.σ), dat) for dat in data )
    return ll
end
objectivesEnzyme = [
    Objective(ModelWrapper(EnzymeBM(), _paramEnzyme, deepcopy(_args), FlattenDefault()), dat),
    Objective(ModelWrapper(EnzymeBM(), _paramEnzyme, deepcopy(_args), FlattenDefault(; output = Float32)), dat)
]
backendsEnzyme = [:EnzymeReverse] #[:EnzymeForward, :EnzymeReverse]
#=
############################################################################################
#!NOTE: MH often does not reach stepsize in default number of steps - keep out in settings
for iter in eachindex(objectivesEnzyme)
    _obj = objectivesEnzyme[iter]
    _flattentype = _obj.model.info.reconstruct.default.output
    @testset "Stepsize initial estimate, all models" begin
        for backend in backendsEnzyme
            mcmcdefault = MCMCDefault(;
                stepsize = ConfigStepsize(; initialstepsize = UpdateTrue(), stepsizeadaption=UpdateTrue()),
                GradientBackend = backend
            )
            ## MCMC kernels
            for kernel in gradientkernels #gradientkernels if error during test
                ## Initialize kernel and check if it can be run
                mcmckernel = MCMC(
                    _rng,
                    kernel,
                    _obj,
                    mcmcdefault
                )
                @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                ## Stepsize Adaption
                _val1, _diag1 = propose(_rng, mcmckernel, _obj)

                ## Check if args and data are not mutated by Enzyme
                @test _obj.model.arg.a == _a
                @test _obj.model.arg.b == _b
                @test _obj.data == dat
                ##
                @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                @test eltype(mcmckernel.tune.proposal.Σ) ==
                    eltype(mcmckernel.tune.proposal.Σ⁻¹ᶜʰᵒˡ) ==
                    eltype(mcmckernel.tune.proposal.chain) ==  _flattentype
            #=
                ## Stepsize Adaption
                proposaltune = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(BaytesCore.Batch(), nothing, nothing) )
                _val2, _diag2 = propose!(_rng, mcmckernel, _obj.model, _obj.data, proposaltune)
                @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                @test eltype(mcmckernel.tune.proposal.Σ) ==
                    eltype(mcmckernel.tune.proposal.Σ⁻¹ᶜʰᵒˡ) ==
                    eltype(mcmckernel.tune.proposal.chain) ==  _flattentype
                ## Stepsize Adaption
                proposaltune = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateFalse(), BaytesCore.DataTune(BaytesCore.Batch(), nothing, nothing) )
                _val3, _diag3 = propose!(_rng, mcmckernel, _obj.model, _obj.data, proposaltune)
                @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                @test eltype(mcmckernel.tune.proposal.Σ) ==
                    eltype(mcmckernel.tune.proposal.Σ⁻¹ᶜʰᵒˡ) ==
                    eltype(mcmckernel.tune.proposal.chain) ==  _flattentype
            =#
            end
        end
    end
end
=#
