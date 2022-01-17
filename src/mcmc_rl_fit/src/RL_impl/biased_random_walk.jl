"""
Biased random walk
------------------

Random walk with a bias for left/right preference.

get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl
"""
function getLL_BiasedRandomWalk(data::Array{Int,2}, bias::Float64)

    data_stats = get_data_stats(data)
    temp = 0.5 # It does not really matter what value we take
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_BiasedRandomWalk supports only one subject")
    end
    nr_states = data_stats.max_state_id
    nr_actions = data_stats.max_action_id
    (nr_samples,) = size(data)
    LL_biased = zeros(Float64, nr_actions)

    # calculate the biased probability for one action
    all_actions = data[:, Int(RL_Fit.c_action)]
    bias_template = zeros(Float64, nr_actions)
    bias_template[1] = bias
    p_action_biased = getSoftmaxProbs(zeros(Float64, nr_actions), temp, bias_template)

    (nr_actions_taken,) = size(data)
    for bias_idx = 1:nr_actions
        a_count = sum(all_actions .== bias_idx)
        others_count = nr_actions_taken - a_count
        # 1 is the index of the biased action. all others (therefore 2) are the unbiased actions
        LL_biased[bias_idx] = a_count * log(p_action_biased[1]) + others_count * log(p_action_biased[2])
    end
    maxLL, biased_action_idx = findmax(LL_biased)

    # set all action probabilities to p_unbiased (index 2 or more), then set the biased one
    action_selection_probs = p_action_biased[2] * ones(Float64, nr_samples)  # all non biased
    i_biased_actions = all_actions .== biased_action_idx

    action_selection_probs[i_biased_actions] .= p_action_biased[1]  # the one biased action

    signals = Step_by_step_signals(
    data_stats.all_subject_ids[1],
    [], #RPE
    [], #SPE
    action_selection_probs,
    maxLL,
    data_stats,
    "Biased random walk",
    [bias],
    "bias=$(round(bias,digits=3))",
    biased_action_idx,
    Dict{String,Any}() # no additional results
    )
    return maxLL, signals
end
