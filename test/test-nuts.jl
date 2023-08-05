#!TODO: Yet to write more tests here

############################################################################################
objectives = [deepcopy(myobjective)]
_depth = [10, 1, 30]
for iter in eachindex(objectives)
    for depth in _depth
        _obj = objectives[iter]
        _flattentype = _obj.model.info.reconstruct.default.output
        @testset "Kernel construction and propagation, all models" begin    
            mcmcdefault = MCMCDefault(;
            kernel = (; max_depth = depth),
            stepsize = ConfigStepsize(;
                stepsizeadaption=UpdateFalse()),
                GradientBackend = backends[1],
                generated = UpdateTrue()
            )
            mcmckernel = MCMC(
                _rng,
                NUTS,
                _obj,
                mcmcdefault
            )
            # Check tree depth
            @test mcmckernel.kernel.max_depth == depth
            # Make proposal depth and check if tree depth not exceeded
            dat = _obj.data
            _valnuts, _diagnuts = propose!(_rng, mcmckernel, _obj.model, dat)
            _diagnuts.kernel.depth
            @test _diagnuts.kernel.depth <= depth
        end
    end
end