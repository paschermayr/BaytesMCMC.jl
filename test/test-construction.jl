############################################################################################
# Models to be used in construction
objectives = [Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault(; output = Float32)), data_uv)
    ]

## Make model
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.flattendefault.output
    @testset "Kernel construction and propagation, all models" begin
        ## MCMC AD backends
        for backend in backends
            ## Mass matrix adaption settings
            for matrixadaption in massmatrices
                ## Define MCMC default tuning parameter
                mcmcdefault = MCMCDefault(;
                    config_kw = (; metric = matrixadaption),
                    GradientBackend = backend)
                ## MCMC kernels
                for kernel in kernels
                    ## Check if Default options work
                    MCMC(kernel, _obj; default = mcmcdefault)
                    MCMC(kernel, _obj, 1; default = mcmcdefault)
                    MCMC(kernel, _obj, 1, TemperDefault(); default = mcmcdefault)
                    ## Check if constructors work
                    constructor = MCMCConstructor(kernel, keys(_obj.tagged.parameter), mcmcdefault)
                    constructor(_rng, _obj.model, _obj.data, 1, BaytesCore.TemperDefault())
                    ## Initialize kernel and check if it can be run
                    mcmckernel = MCMC(
                        _rng,
                        kernel,
                        _obj,
                        1,
                        BaytesCore.TemperDefault(BaytesCore.UpdateTrue(), 0.5);
                        default = mcmcdefault
                    )
                    propose(_rng, mcmckernel, _obj)
                    ## Test if Float types for proposal calculation all have same type after proposal step
                    # !NOTE kernel.result structs are already checked in ModelWrappers
                    @test mcmckernel.tune.stepsize.ϵ isa _flattentype
                    @test mcmckernel.tune.tempering.val.current isa _flattentype
                    @test eltype(mcmckernel.tune.proposal.Σ) ==
                        eltype(mcmckernel.tune.proposal.Σ⁻¹ᶜʰᵒˡ) ==
                        eltype(mcmckernel.tune.proposal.chain) ==  _flattentype
                end
            end
        end
    end
end
