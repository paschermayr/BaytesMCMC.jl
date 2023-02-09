#!TODO: Yet to write more tests here

############################################################################################
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.reconstruct.default.output
    @testset "Kernel construction and propagation, all models" begin
        mcmcdefault = MCMCDefault(;
        stepsize = ConfigStepsize(;
            stepsizeadaption=UpdateFalse()),
            GradientBackend = backends[1],
            generated = UpdateTrue()
        )
        mcmckernel = MCMC(
            _rng,
            kernel,
            _obj,
            mcmcdefault
        )
    end
end
