"""
REINFORCE learner (Williams 1992) with baseline
(state values V)

See also REINFORCElearner.jl

Specifically for 2 actions and reward only at the end of episode!

Note
----
The policy parameters θ as well as the feature vector ϕ are
column vectors. Here we use a (states x actions) matrix for simpler access.
also, the feature vector ϕ is a one-hot vector in our case. therefore we
do not need it to keep a vector but access the entries by the
state/action pair [s_t, a_t]
"""

mutable struct REINFORCEbaseline
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
    V_baseline::Array{Float64, 1}
    α_baseline::Float64
    do_debug_print::Bool
end
function REINFORCEbaseline(; nr_states = 10, nr_actions = 4, α = 0.1, γ = .9, # initvalue = 0.,
    α_baseline = α, do_debug_print = false)
    θ = zeros(Float64, nr_states, nr_actions) #.+ initvalue,
    V_baseline = zeros(Float64, nr_states)

    REINFORCEbaseline(nr_states, nr_actions, α, α, γ, 1., [], [], [], θ, [0.], [],
                    V_baseline, α_baseline, do_debug_print)
end
export REINFORCEbaseline
function gettarget!(learner::REINFORCEbaseline, state, reward)
    V_s = reward * learner.γ_k  #  note: assuming a single reward, set during the forward pass above
    TD_error = V_s - learner.V_baseline[state]
    learner.V_baseline[state] += learner.α_baseline * TD_error
    learner.target[1] = copy(TD_error)
end
