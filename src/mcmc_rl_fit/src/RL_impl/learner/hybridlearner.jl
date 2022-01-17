"""
Hybrid learner (core)
Non-convex combination of MB and MF Q-values or MF policy parameters
"""

mutable struct Hybrid
    nr_states::Int
    nr_actions::Int
    w_MB::Float64
    w_MF::Float64
    QSA::Array{Float64, 2}
    do_debug_print::Bool
end
function Hybrid(; nr_states = 10, nr_actions = 4,
    w_MB = 3., w_MF = 3.,
    QSA = zeros(Float64, nr_states, nr_actions),
    do_debug_print = false)

    Hybrid(nr_states, nr_actions, w_MB, w_MF, QSA,
            do_debug_print)
end
export Hybrid
function do_debug_print_final(learner::Union{Hybrid, HybridConvex})
    println("Final:")
    for s in 1:size(learner.QSA,1)
        @show learner.QSA[s, :]
    end
    println("--------------")
end
""" Update: main function """
function update!(learner::Union{Hybrid, HybridConvex}, state, action, reward, QSA_MB, QSA_MF)
    gethybridweights!(learner)
    learner.QSA =  learner.w_MB*QSA_MB + learner.w_MF*QSA_MF
    updatetimecounter!(learner, reward)
    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!
gethybridweights!(learner::Hybrid) = nothing
updatetimecounter!(learner::Hybrid, reward) = nothing
