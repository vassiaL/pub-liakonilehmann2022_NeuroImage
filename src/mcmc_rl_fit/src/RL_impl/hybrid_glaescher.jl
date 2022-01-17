"""

# Hybrid-0 (Glaescher et al. 2010)
# --------------------------------

Hybrid: Forward Learner + SARSA-0

See learner/forwardlearner.jl, learner/sarsaLlearner.jl
and learner/hybridconvexlearner.jl for the core implementation.

"""
function getLL_GlaescherHybrid(data::Array{Int,2}, alpha::Float64, gamma::Float64,
                                eta::Float64, decay_offset::Float64,
                                decay_slope::Float64, temp::Float64)
    maxLL, signals = getLL_HybridSarsaLambda(data, alpha, gamma::Float64,
                                    0., eta, decay_offset,
                                    decay_slope, temp)
    algo = "Hybrid Glaescher FWD+Sarsa)"
    param_v = [alpha, gamma, eta, decay_offset, decay_slope, temp]
    param_str = "alpha=$(round(alpha,digits=4)), gamma=$(round(gamma,digits=4)),
                eta=$(round(eta,digits=4)), decay_offset=$(round(decay_offset,digits=4)),
                decay_slope=$(round(decay_slope,digits=4)), temp=$(round(temp,digits=4))"
    s = Step_by_step_signals(
        signals.subject_id,
        signals.RPE,
        signals.SPE,
        signals.action_selection_prob,
        signals.LL,
        signals.data_stats,
        algo,
        param_v,
        param_str,
        signals.max_bias_action_index,
        signals.results_dict
        )
    return maxLL, s
end
