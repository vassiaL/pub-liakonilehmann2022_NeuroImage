"""
Hybrid learner (core)
Convex combination of MB and MF Q-values
Time decaying (or increasing) MB weights
"""
mutable struct HybridConvex
    nr_states::Int
    nr_actions::Int
    w_MB::Float64
    w_MF::Float64
    t_hybrid::Float64
    decay_offset::Float64
    decay_slope::Float64
    QSA::Array{Float64, 2}
    do_debug_print::Bool
end
function HybridConvex(; nr_states = 10, nr_actions = 4,
    decay_offset = 0.1, decay_slope = 0.1,
    QSA = zeros(Float64, nr_states, nr_actions),
    do_debug_print = false)
    t_hybrid = 0.
    w_MB = get_weight(decay_offset, decay_slope, t_hybrid)
    w_MF = 1. - w_MB

    HybridConvex(nr_states, nr_actions, w_MB, w_MF, t_hybrid, decay_offset, decay_slope,
                QSA, do_debug_print)
end
export HybridConvex
function gethybridweights!(learner::HybridConvex)
    learner.w_MB = get_weight(learner.decay_offset, learner.decay_slope, learner.t_hybrid)
    learner.w_MF = 1. - learner.w_MB
end
function updatetimecounter!(learner::HybridConvex, reward)
    if reward>0 #goal state, reward, NO NEXT ACTION (in our paradigm)
        learner.t_hybrid += 1
    end
end
function get_weight(w0, T, t)
    # compute a weight that can decrease to 0 or grow to 1 (unlike original Glascher paper)
    epsilon = 0.01
    if T>epsilon # have a small margin, avoiding division by
        tau = 1.0 / T
        return w0 * exp(-t/tau)
    elseif T >= -epsilon
        return w0
    else
        tau = -1.0 / T
        return 1.0 - (1.0 -w0) * exp(-t/tau)
    end
end
