"""
REINFORCE learner (Williams 1992)
with a suprise-modulated learning rate.

See also REINFORCElearner.jl and sarsaLmodulatedlearner.jl

Specifically for 2 actions and reward only at the end of episode!

Note
----
The policy parameters θ as well as the feature vector ϕ are
column vectors. Here we use a (states x actions) matrix for simpler access.
also, the feature vector ϕ is a one-hot vector in our case. therefore we
do not need it to keep a vector but access the entries by the
state/action pair [s_t, a_t]
"""

mutable struct REINFORCEModulated
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
function REINFORCEModulated(; nr_states = 10, nr_actions = 4, α = 0.1, γ = .9,
    do_debug_print = false)
    θ = zeros(Float64, nr_states, nr_actions)

    REINFORCEModulated(nr_states, nr_actions, α, α, γ, 1., [], [], [], θ, [0.],
                        [], do_debug_print)
end
export REINFORCEModulated
