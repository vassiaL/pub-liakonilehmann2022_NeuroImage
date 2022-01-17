# Includes all algorithms
# -----------------------

include("policies.jl")
include("biased_random_walk.jl")

include("sarsaL.jl")
include("sarsaL_modulated_particlefilterjump.jl")

include("forward.jl")
include("hybrid_sarsaL.jl")
include("hybrid_glaescher.jl")

include("REINFORCE.jl")
include("REINFORCE_modulated.jl")
include("REINFORCE_baseline.jl")
include("actor_critic.jl")
include("actor_critic_modulatedActor.jl")

include(joinpath("learner", "Tlearner", "tgeneric.jl"))
include(joinpath("learner", "Tlearner", "tleakyglaescher.jl"))
include(joinpath("learner", "Tlearner", "tparticlefilterjump.jl"))

include(joinpath("learner", "Rlearner", "rewardlearners.jl"))

include(joinpath("learner", "prioritizedsweepinglearner.jl"))
include(joinpath("learner", "REINFORCEmodulatedlearner.jl"))
include(joinpath("learner", "REINFORCEbaselinelearner.jl"))
include(joinpath("learner", "REINFORCElearner.jl"))
include(joinpath("learner", "sarsaLmodulatedlearner.jl"))
include(joinpath("learner", "sarsaLlearner.jl"))
include(joinpath("learner", "actorcriticmodulatedlearner.jl"))
include(joinpath("learner", "actorcriticlearner.jl"))
include(joinpath("learner", "hybridconvexlearner.jl"))
include(joinpath("learner", "hybridlearner.jl"))
include(joinpath("learner", "forwardlearnerBF.jl"))
include(joinpath("learner", "forwardlearner.jl"))


include("prioritized_sweeping_particlefilterjump.jl")
include("hybrid_prioritizedsweeping_particlefilterjump.jl")
include("hybrid_actorcritic_FWD.jl")
include("hybrid_actorcritic_FWD_BF.jl")

include("actor_critic_actioncost.jl")
include("actor_critic_modulatedActor_actioncost.jl")
