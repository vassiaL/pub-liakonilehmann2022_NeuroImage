"""
REINFORCE learner (Williams 1992)

Specifically for 2 actions and reward only at the end of episode!

Note
----
The policy parameters θ as well as the feature vector ϕ are
column vectors. Here we use a (states x actions) matrix for simpler access.
also, the feature vector ϕ is a one-hot vector in our case. therefore we
do not need it to keep a vector but access the entries by the
state/action pair [s_t, a_t]
"""
mutable struct REINFORCE
    nr_states::Int
    nr_actions::Int
    α::Float64
    α_eff::Float64
    γ::Float64
    γ_k::Float64
    action_selection_prob_in_episode::Array{Any,1}
    states_in_episode::Array{Any,1}
    actions_in_episode::Array{Any,1}
    θ::Array{Float64, 2}
    target::Array{Float64, 1}
    modulatingfactor_in_episode::Array{Any,1}
    do_debug_print::Bool
end
function REINFORCE(; nr_states = 10, nr_actions = 4, α = 0.1, γ = .9,
    do_debug_print = false)
    θ = zeros(Float64, nr_states, nr_actions)

    REINFORCE(nr_states, nr_actions, α, α, γ, 1., [], [], [], θ, [0.], [], do_debug_print)
end
export REINFORCE
function do_debug_print_final(learner::Union{REINFORCE, REINFORCEModulated, REINFORCEbaseline})
    println("Final:")
    for s in 1:size(learner.θ,1)
        @show learner.θ[s, :]
    end
    println("--------------")
end
""" Update: main function """
function update!(learner::Union{REINFORCE, REINFORCEModulated, REINFORCEbaseline},
                state, action, reward, p_action, modulatingfactor)
    push!(learner.action_selection_prob_in_episode, p_action)
    push!(learner.states_in_episode, state)
    push!(learner.actions_in_episode, action)
    push!(learner.modulatingfactor_in_episode, modulatingfactor)
    if reward>0
         # Backward pass to update the policy
        for k = (length(learner.states_in_episode) : -1 : 1)

            getlearningrate!(learner, learner.modulatingfactor_in_episode[k])

            gettarget!(learner, learner.states_in_episode[k], reward)

            learner.θ[learner.states_in_episode[k], learner.actions_in_episode[k]] +=
                        learner.α_eff * learner.target[1] * (1. - learner.action_selection_prob_in_episode[k])
            learner.θ[learner.states_in_episode[k], 3-learner.actions_in_episode[k]] +=
                        learner.α_eff * learner.target[1] * (0. - (1. - learner.action_selection_prob_in_episode[k]))
            learner.γ_k *= learner.γ
        end
        # Reset
        learner.γ_k = 1.
        learner.action_selection_prob_in_episode = []
        learner.states_in_episode = []
        learner.actions_in_episode = []
        learner.modulatingfactor_in_episode = []
    end
    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!
function gettarget!(learner::Union{REINFORCE, REINFORCEModulated}, state, reward)
    learner.target[1] = reward * learner.γ_k  #  note: assuming a single reward, set during the forward pass above
end
