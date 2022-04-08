#!TODO: Yet to write more tests here

#=
Disclaimer: Many of the tests in this file taken from:
    https://github.com/tpapp/DynamicHMC.jl/blob/master/test/test-NUTS.jl
=#

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

############################################################################################
@testset "random booleans" begin
    @test rand_bool_logprob(_rng, 0)
    @test !rand_bool_logprob(_rng, log(0))
end
