"""
Actor-Critic learner (core)

Critic implemented by td-lambda
This implementation follows Sutton & Barto,
Chapter 7.7 "Eligibility Traces for Actor-Critic Methods"
The policy update follows the scheme given in chapter 6.6 where
the policy changes inversely proportional to the action selection probability
"""

mutable struct ActorCritic
    nr_states::Int
    nr_actions::Int
    α_critic::Float64
    α_actor::Float64
    α_actor_eff::Float64
    γ::Float64
    λ_critic::Float64
    λ_actor::Float64
    initvalue::Float64
    V::Array{Float64, 1}
    policyparameters::Array{Float64, 2}
    eligTrace_critic::Array{Float64, 1}
    eligTrace_actor::Array{Float64, 2}
    td_error::Array{Float64, 1}
    use_replacing_trace::Bool
    do_debug_print::Bool
    actioncost::Float64
end
function ActorCritic(; nr_states = 10, nr_actions = 4,
    α_critic = 0.1, α_actor = 0.1, γ = .9,
    λ_critic = 0.9, λ_actor = 0.9, initvalue = 0.,
    V = zeros(Float64, nr_states) .+ initvalue,
    policyparameters = zeros(Float64, nr_states, nr_actions),
    eligTrace_critic = zeros(Float64, nr_states),
    eligTrace_actor = zeros(Float64, nr_states, nr_actions),
    use_replacing_trace = true,
    do_debug_print = false,
    actioncost = 0.)

    ActorCritic(nr_states, nr_actions, α_critic, α_actor, α_actor, γ,
        λ_critic, λ_actor, initvalue, V,
        policyparameters, eligTrace_critic, eligTrace_actor, [0.],
        use_replacing_trace, do_debug_print, actioncost)
end
export ActorCritic
function do_debug_print_final(learner::Union{ActorCritic, ActorCriticModulated})
    println("Final:")
    for s in 1:size(learner.V,1)
        @show learner.V[s]
    end
    for s in 1:size(learner.policyparameters,1)
        @show learner.policyparameters[s,:]
    end
    println("--------------")
end

""" Update: main function """
function update!(learner::Union{ActorCritic, ActorCriticModulated},
                state, action, next_state, reward,
                p_action, modulatingfactor)

    if learner.use_replacing_trace # replacing trace
        learner.eligTrace_critic[state] = 1.
        learner.eligTrace_actor[state, action] = (1. - p_action) #see eq. 7.12/7.13 in Sutton & Barto
    else # (accumulating) eligibility trace
        learner.eligTrace[state] += 1.
        learner.eligTrace_actor[state, action] += (1. - p_action) #see eq. 7.12 in Sutton & Barto
    end

    getlearningrate!(learner, modulatingfactor)

    learner.td_error[1] = reward - learner.actioncost + (learner.γ * learner.V[next_state]) - learner.V[state]
    learner.V[:] .+= learner.α_critic  * learner.td_error[1] * learner.eligTrace_critic
    learner.policyparameters[:, :] .+= learner.α_actor_eff * learner.td_error[1] * learner.eligTrace_actor

    if reward>0
        learner.eligTrace_critic = zeros(Float64, learner.nr_states)
        learner.eligTrace_actor = zeros(Float64, learner.nr_states, learner.nr_actions)
    else #non goal state
        learner.eligTrace_actor[:, :] .*= learner.λ_actor * learner.γ
        learner.eligTrace_critic[:] .*= learner.λ_critic * learner.γ
    end
    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!
function getlearningrate!(learner::ActorCritic, modulatingfactor)
    learner.α_actor_eff = learner.α_actor
end
