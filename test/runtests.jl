############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!
using UnPack: UnPack, @unpack, @pack!
using Distributions

############################################################################################
# Import Baytes Packages
using BaytesCore, ModelWrappers, BaytesMCMC
include("E:\\OneDrive\\1_Professional\\1_Git\\0_Dev\\Julia\\modules\\BaytesMCMC.jl\\src\\BaytesMCMC.jl")
using .BaytesMCMC
############################################################################################
# Include Files
include("TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-construction.jl")
end
