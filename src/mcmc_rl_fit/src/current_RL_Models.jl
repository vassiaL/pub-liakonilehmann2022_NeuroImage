"""
Main module for learning algorithms and behavioral fitting.

All algorithms are 'included' via the file RL_impl/discrete_RL_replay.jl

Also here:
Wrapper functions that makes each algorithm implementation callable from within
our MCMC fitting framework.

For each algorithm there is a wrapper function here below (getLL_***_V).
Each getLL_***_V calls the corresponding getLL_*** of the algorithm.

For example, for the Surprise Actor-critic (actor_critic_modulatedActor.jl)
the corresponding wrapper function is getLL_ActorCritic_TDLambda_modulatedActor_V,
which calls the getLL_ActorCritic_TDLambda_modulatedActor
in actor_critic_modulatedActor.jl
"""
module RL_Fit

using Distributions
using DataStructures
using Distributed
using Random

include(joinpath("utils", "params.jl"))
include(joinpath("utils", "rl_data.jl"))
include(joinpath("utils", "rl_data_toolsforalgos.jl"))
include(joinpath("utils","visualizations.jl"))
include(joinpath("utils","metropolis_fit.jl"))
include(joinpath("utils","crossvalidation_fit.jl"))
include(joinpath("utils","get_fit_tools.jl"))
include(joinpath("utils","get_meanLL_acrossseeds.jl"))

# discrete_RL_replay = file that includes all algorithms
include(joinpath("RL_impl","discrete_RL_replay.jl"))

include("sinergia_fit_tools.jl")

println("Loading files current_RL_Models.jl and discrete_RL_replay.jl")

# Test parallelization
function sayHi()
     println("module RL_Fit says hi from worker id = $(myid())")
end

# ------------------------------------------------------------------------------
# ----------------- Wrapper functions ------------------------------------------
# ------------------------------------------------------------------------------


# Surprise SARSA-λ
# ----------------
function getLL_sarsaL_modulated_TParticleFilterJump_V(data::Array{Int,2},
                                                paramVector;
                                                particlefilterseeds = [1,2,3]
                                                )
    # nseeds = 3
    # particlefilterseeds = rand(1:typemax(UInt64)-1, 3)
    learner_function(subID, d) = getLL_sarsaL_modulated_TParticleFilterJump(d,
                                                                paramVector[1], #gamma
                                                                paramVector[2], #temp
                                                                paramVector[3], #surpriseprobability
                                                                paramVector[4], #stochasticity
                                                                paramVector[5], #nparticles
                                                                paramVector[6], #alpha_0
                                                                paramVector[7], #lambda
                                                                particlefilterseeds = particlefilterseeds
                                                                )
    generic_LL_V(data, learner_function)
end

# PS with PF
# ----------
function getLL_Prio_Sweep_TParticleFilterJump_V(data::Array{Int,2}, paramVector ;
                                                particlefilterseeds = [1,2,3]
                                                )
    # nseeds = 3
    # particlefilterseeds = rand(1:typemax(UInt64)-1, 3)
    learner_function(subID, d) = getLL_Prio_Sweep_TParticleFilterJump(d,
                                                paramVector[1], # gamma
                                                paramVector[2], # temp
                                                paramVector[3], # surpriseprobability
                                                paramVector[4], # stochasticity
                                                paramVector[5], #nr_update_cycles
                                                paramVector[6], #nparticles
                                                particlefilterseeds = particlefilterseeds
                                                )
    generic_LL_V(data, learner_function)
end

# Hybrid-λ-PS-PF
# --------------
function getLL_Hybrid_Prio_Sweep_TParticleFilterJump_V(data::Array{Int,2},
                                                paramVector ;
                                                particlefilterseeds = [1,2,3]
                                                )

    learner_function(subID, d) = getLL_Hybrid_Prio_Sweep_TParticleFilterJump(d,
                                                                paramVector[1], #gamma
                                                                paramVector[2], #surpriseprobability
                                                                paramVector[3], #stochasticity
                                                                paramVector[4], #nr_upd_cycles
                                                                paramVector[5], #nparticles
                                                                paramVector[6], #alpha
                                                                paramVector[7], #lambda
                                                                paramVector[8], #w_MB
                                                                paramVector[9], #w_MF
                                                                particlefilterseeds = particlefilterseeds
                                                                )
    generic_LL_V(data, learner_function)
end

# Random Walk with a bias
# -----------------------
function getLL_BiasedRandomWalk_V(data::Array{Int,2}, paramVector)
    learner_function(subID, d) = getLL_BiasedRandomWalk(d, paramVector[1])
    generic_LL_V(data, learner_function)
end

# SARSA-λ
# -------
function getLL_SarsaLambda_V(data::Array{Int,2}, paramVector, initQSA...)
    # Sarsa with Replacing Traces (default)
    if length(initQSA) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_SarsaLambda(d, paramVector[1], paramVector[2], paramVector[3], paramVector[4]))
    else
        # println("got initial QSA: $initQSA")
        generic_LL_V(data, (subID, d) -> getLL_SarsaLambda(d, paramVector[1], paramVector[2], paramVector[3], paramVector[4]; initQSA=initQSA[1][subID]))
    end
end

# Hybrid-0 (Glaescher et al. 2010)
# --------------------------------
function getLL_GlaescherHybrid_V(data::Array{Int,2}, paramVector)
    learner_function(subID, d) = getLL_GlaescherHybrid(d, paramVector[1], paramVector[2],
                                    paramVector[3], paramVector[4], paramVector[5], paramVector[6])
    generic_LL_V(data, learner_function)
end

# Hybrid-λ (Daw et al. 2011)
# --------------------------
function getLL_HybridSarsaLambda_V(data::Array{Int,2}, paramVector)
    learner_function(subID, d) = getLL_HybridSarsaLambda(d, paramVector[1], paramVector[2],
                                    paramVector[3], paramVector[4], paramVector[5],
                                    paramVector[6], paramVector[7])
    generic_LL_V(data, learner_function)
end

# FWD (Glaescher et al. 2010)
# ---------------------------
function getLL_Forward_Learner_V(data::Array{Int,2}, paramVector, initTsasR...)
    if length(initTsasR) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_Forward_Learner(d, paramVector[1],
                                        paramVector[2], paramVector[3]))
    else
        # println("got initTsasR: $initTsasR")
        generic_LL_V(data, (subID, d) -> getLL_Forward_Learner(d, paramVector[1],
                                        paramVector[2], paramVector[3]; initTsas = initTsasR[1][subID], initR = initTsasR[2][subID]))
    end
end

# REINFORCE
# ---------
function getLL_REINFORCE_V(data::Array{Int,2}, paramVector)
    learner_function(subID, d)= getLL_REINFORCE(d, paramVector[1], paramVector[2], paramVector[3])
    generic_LL_V(data, learner_function)
end

# Surprise-REINFORCE
# ------------------
function getLL_REINFORCE_modulated_V(data::Array{Int,2}, paramVector;
                                    particlefilterseeds = [1,2,3])

    learner_function(subID, d)= getLL_REINFORCE_modulated(d,
                                                        paramVector[1], #alpha
                                                        paramVector[2], #surpriseprobability
                                                        paramVector[3], #stochasticity
                                                        paramVector[4], #gamma
                                                        paramVector[5], #temp
                                                        paramVector[6], #nparticles
                                                        particlefilterseeds = particlefilterseeds
                                                        )

    generic_LL_V(data, learner_function)
end

# REINFORCE baseline
# ------------------
function getLL_REINFORCE_baseline_V(data::Array{Int,2}, paramVector)
    learner_function(subID, d)= getLL_REINFORCE_baseline(d, paramVector[1], #alpha
                                                            paramVector[2], #gamma
                                                            paramVector[3], #temp
                                                            paramVector[4] # alpha_baseline
                                                            )
    generic_LL_V(data, learner_function)
end

# Actor-critic
# ------------
function getLL_ActorCritic_TDLambda_V(data::Array{Int,2}, paramVector, initThetaVs...)
    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda(d, paramVector[1], paramVector[2], paramVector[3], paramVector[4], paramVector[5], paramVector[6]))
    else
        # println("got initThetaVs: $initThetaVs")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda(d, paramVector[1], paramVector[2], paramVector[3], paramVector[4], paramVector[5], paramVector[6]; init_policy=initThetaVs[1][subID], init_Vs = initThetaVs[2][subID]))
    end
end

# Surprise Actor-critic
# ---------------------
function getLL_ActorCritic_TDLambda_modulatedActor_V(data::Array{Int,2}, paramVector, initThetaVs...)
    particlefilterseeds = [1,2,3]

    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_modulatedActor(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #surpriseprobability
                                    paramVector[3], #stochasticity
                                    paramVector[4], #eta
                                    paramVector[5], #gamma
                                    paramVector[6], #lambda_actor
                                    paramVector[7], #lambda_critic
                                    paramVector[8], #temp
                                    paramVector[9], #nparticles
                                    particlefilterseeds = particlefilterseeds,
                                    ))
    else
        # println("got initThetaVs: $initThetaVs")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_modulatedActor(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #surpriseprobability
                                    paramVector[3], #stochasticity
                                    paramVector[4], #eta
                                    paramVector[5], #gamma
                                    paramVector[6], #lambda_actor
                                    paramVector[7], #lambda_critic
                                    paramVector[8], #temp
                                    paramVector[9], #nparticles
                                    particlefilterseeds = particlefilterseeds,
                                    init_policy=initThetaVs[1][subID],
                                    init_Vs = initThetaVs[2][subID]))
    end
end

# Hybrid Actor-critic
# -------------------
function getLL_Hybrid_AC_FWD_V(data::Array{Int,2}, paramVector, initThetaVs...)
    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        generic_LL_V(data, (subID, d) -> getLL_Hybrid_AC_FWD(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #eta_critic
                                    paramVector[3], #gamma
                                    paramVector[4], #lambda_actor
                                    paramVector[5], #lambda_critic
                                    paramVector[6], #eta_fwd
                                    paramVector[7], #w_MB
                                    paramVector[8], #w_MF
                                    ))
    else
        generic_LL_V(data, (subID, d) -> getLL_Hybrid_AC_FWD(d,
                                    pparamVector[1], #alpha
                                    paramVector[2], #eta_critic
                                    paramVector[3], #gamma
                                    paramVector[4], #lambda_actor
                                    paramVector[5], #lambda_critic
                                    paramVector[6], #eta_fwd
                                    paramVector[7], #w_MB
                                    paramVector[8], #w_MF
                                    init_policy=initThetaVs[1][subID],
                                    init_Vs = initThetaVs[2][subID]))
    end
end

# Hybrid Actor-critic PF
# ----------------------
function getLL_Hybrid_AC_FWD_BF_V(data::Array{Int,2}, paramVector, initThetaVs...)
    particlefilterseeds = [1,2,3]

    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        generic_LL_V(data, (subID, d) -> getLL_Hybrid_AC_FWD_BF(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #eta_critic
                                    paramVector[3], #gamma
                                    paramVector[4], #lambda_actor
                                    paramVector[5], #lambda_critic
                                    paramVector[6], #surpriseprobability
                                    paramVector[7], #stochasticity
                                    paramVector[8], #nparticles
                                    paramVector[9], #w_MB
                                    paramVector[10], #w_MF
                                    particlefilterseeds = particlefilterseeds
                                    ))
    else
        generic_LL_V(data, (subID, d) -> getLL_Hybrid_AC_FWD_BF(d,
                                    pparamVector[1], #alpha
                                    paramVector[2], #eta_critic
                                    paramVector[3], #gamma
                                    paramVector[4], #lambda_actor
                                    paramVector[5], #lambda_critic
                                    paramVector[6], #surpriseprobability
                                    paramVector[7], #stochasticity
                                    paramVector[8], #nparticles
                                    paramVector[9], #w_MB
                                    paramVector[10], #w_MF
                                    particlefilterseeds = particlefilterseeds,
                                    init_policy=initThetaVs[1][subID],
                                    init_Vs = initThetaVs[2][subID]))
    end
end

# Actor-critic-cost
# -----------------
function getLL_ActorCritic_TDLambda_actioncost_V(data::Array{Int,2}, paramVector, initThetaVs...)
    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_actioncost(d, paramVector[1], paramVector[2],
                                                                    paramVector[3], paramVector[4], paramVector[5],
                                                                    paramVector[6], paramVector[7]))
    else
        # println("got initThetaVs: $initThetaVs")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_actioncost(d, paramVector[1], paramVector[2],
                                                                    paramVector[3], paramVector[4], paramVector[5],
                                                                    paramVector[6], paramVector[7];
                                                                    init_policy=initThetaVs[1][subID], init_Vs = initThetaVs[2][subID]))
    end
end

# Surprise Actor-critic-cost
# --------------------------
function getLL_ActorCritic_TDLambda_modulatedActor_actioncost_V(data::Array{Int,2}, paramVector, initThetaVs...)
    particlefilterseeds = [1,2,3]

    # AC with Replacing Traces (default)
    if length(initThetaVs) == 0
        # println("no init value")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_modulatedActor_actioncost(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #surpriseprobability
                                    paramVector[3], #stochasticity
                                    paramVector[4], #eta
                                    paramVector[5], #gamma
                                    paramVector[6], #lambda_actor
                                    paramVector[7], #lambda_critic
                                    paramVector[8], #temp
                                    paramVector[9], #nparticles
                                    paramVector[10], #actioncost
                                    particlefilterseeds = particlefilterseeds,
                                    ))
    else
        # println("got initThetaVs: $initThetaVs")
        generic_LL_V(data, (subID, d) -> getLL_ActorCritic_TDLambda_modulatedActor_actioncost(d,
                                    paramVector[1], #alpha
                                    paramVector[2], #surpriseprobability
                                    paramVector[3], #stochasticity
                                    paramVector[4], #eta
                                    paramVector[5], #gamma
                                    paramVector[6], #lambda_actor
                                    paramVector[7], #lambda_critic
                                    paramVector[8], #temp
                                    paramVector[9], #nparticles
                                    paramVector[10], #actioncost
                                    particlefilterseeds = particlefilterseeds,
                                    init_policy=initThetaVs[1][subID],
                                    init_Vs = initThetaVs[2][subID]))
    end
end

# ----- end of module
end
