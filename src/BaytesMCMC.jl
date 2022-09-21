module BaytesMCMC

################################################################################
#Import modules
using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractKernel,
    AbstractConstructor,
    Updater,
    Iterator,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    logsumexp,
    logaddexp,
    logmeanexp,
    update,
    AcceptStatistic,
    SampleDefault,
    to_Tuple,
    ProposalTune

import BaytesCore:
    BaytesCore,
    update,
    update!,
    infer,
    results,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    result!,
    get_result,
    generate_showvalues

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective,
    sample,
    sample!

import ModelWrappers:
    ModelWrappers,
    predict,
    generate,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization

using BaytesDiff:
    BaytesDiff,
    DiffObjective,
    AbstractDifferentiableTune,
    ℓObjectiveResult,
    ℓDensityResult,
    ℓGradientResult,
    checkfinite,
    AutomaticDiffTune,
    AnalyticalDiffTune

import BaytesDiff:
    BaytesDiff,
    checkfinite

using Random: Random, AbstractRNG, GLOBAL_RNG, randexp
using LinearAlgebra:
    LinearAlgebra, Symmetric, Diagonal, diag, cholesky, inv, UniformScaling, dot, det, I
using Statistics: Statistics, mean, median, std, sqrt, quantile, var, cor, cov, middle
using Distributions: Distributions, logpdf, MvNormal

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!

################################################################################
#Abstract types to be dispatched in Examples section
abstract type MCMCKernel <: AbstractKernel end
abstract type MCMCKernelDiagnostics <: AbstractDiagnostics end
function get_acceptrate() end
function move() end

include("Core/Core.jl")
include("Kernels/Kernels.jl")

################################################################################
export
    # BaytesCore
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    propose,
    propose!,
    propagate,
    propagate!,
    results,
    update!,
    SampleDefault,

    # ModelWrappers
    init,
    init!,
    predict,
    generate,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization,

    #BaytesDiff
    checkfinite,

    #MCMC
    MCMCKernel,
    MCMCKernelDiagnostics,
    MCMCDiagnostics,
    get_acceptrate,
    move

end
