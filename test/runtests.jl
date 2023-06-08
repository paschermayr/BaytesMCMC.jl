############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!
using SimpleUnPack: SimpleUnPack, @unpack, @pack!
using Distributions

############################################################################################
# Import Baytes Packages
using BaytesCore, ModelWrappers, BaytesMCMC
#using .BaytesMCMC
############################################################################################
# Include Files
include("TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-construction.jl")
    #include("test-nuts.jl")
end
