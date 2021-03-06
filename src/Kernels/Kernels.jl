"
Working Examples for this module
"
############################################################################################
#Import
include("utility/utility.jl")
include("Metropolis/Metropolis.jl")
include("MALA/MALA.jl")
include("HMC/HMC.jl")
include("NUTS/NUTS.jl")
#!NOTE: A kernel that ignores MCMC tuning and explicitly takes user proposal distribution.
include("Custom/Custom.jl")

############################################################################################
#Export ~ All MCMC algorithm need to be dispatched on the following functions:
export update!, init, propagate, get_acceptrate
