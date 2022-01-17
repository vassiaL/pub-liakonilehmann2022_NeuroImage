"""
Actor-Critic with a surprise-modulated learning rate for the actor (core)

See also actorcriticlearner.jl

Critic implemented by td-lambda
This implementation follows Sutton & Barto,
Chapter 7.7 "Eligibility Traces for Actor-Critic Methods"
The policy update follows the scheme given in chapter 6.6 where
the policy changes inversely proportional to the action selection probability
"""
mutable struct ActorCriticModulated
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
function ActorCriticModulated(; nr_states = 10, nr_actions = 4,
    α_critic = 0.1, α_actor = 0.1, γ = .9,
    λ_critic = 0.9, λ_actor = 0.9, initvalue = 0.,
    V = zeros(Float64, nr_states) .+ initvalue,
    policyparameters = zeros(Float64, nr_states, nr_actions),
    eligTrace_critic = zeros(Float64, nr_states),
    eligTrace_actor = zeros(Float64, nr_states, nr_actions),
    use_replacing_trace = true,
    do_debug_print = false,
    actioncost = 0.)

    ActorCriticModulated(nr_states, nr_actions, α_critic, α_actor, α_actor, γ,
        λ_critic, λ_actor, initvalue, V,
        policyparameters, eligTrace_critic, eligTrace_actor, [0.],
        use_replacing_trace, do_debug_print, actioncost)
end
export ActorCriticModulated
function getlearningrate!(learner::ActorCriticModulated,
                        modulatingfactor)
    learner.α_actor_eff = learner.α_actor * modulatingfactor
end
