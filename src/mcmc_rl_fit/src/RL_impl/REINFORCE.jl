"""
REINFORCE
---------

Williams, 1992

This implementation follows the pseudo code given in
Sutton Barto, "RL an Intro", Chapter 13 Monte Carlo Policy Gradient.

The policy parameters θ as well as the feature vector ϕ are
column vectors. Here we use a states x actions matrix for simpler access.
Also, the feature vector ϕ is a one-hot vector in our case. Therefore we
do not need to keep a vector but access the entries by the
state/action pair [s_t, a_t]

See learner/REINFORCElearner.jl for the core implementation.

Note
----
get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl
"""
function getLL_REINFORCE(data::Array{Int,2}, alpha::Float64,
                        gamma::Float64, temp::Float64; do_debug_print = false)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_REINFORCE supports only one subject")
    end
    if nr_actions!=2; error("REINFORCE supports two actions only") end
    all_puzzles = get_all_puzzles(data)

    LL_unbiased = 0.
    action_counter = 0
    probs_unbiased = zeros(Float64, nr_samples)
    Δpolicy_record = zeros(Float64, nr_samples)

    PolicyParametersImbalance_record = zeros(Float64, nr_samples)
    PolicyImbalance_record = zeros(Float64, nr_samples)
    PolicyParametersImbalanceAllstates_record = Array{Array{Float64,1}, 1}(undef, nr_samples)
    [PolicyParametersImbalanceAllstates_record[i] = zeros(Float64, nr_states) for i in 1:nr_samples]
    PolicyParametersDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)
    PolicyDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)

    modulatingfactor = 1.
    learner = []
    for i_puzzle in all_puzzles
        # Puzzle = independent block of the experiment.
        # The code was made for general purposes, in this experiment we
        # have only ONE puzzle.
        learner = REINFORCE(nr_states = nr_states,
                        nr_actions = nr_actions,
                        α = alpha,
                        γ = gamma,
                        do_debug_print = do_debug_print)

        all_epi_in_puzzle = get_all_episodes_in_puzzle(data, i_puzzle)
        for i_episode in all_epi_in_puzzle
            episode_data = get_data_by_puzzle_episode(data, i_puzzle, i_episode)
            path_length = size(episode_data)[1]  # = T
            if do_debug_print
                println("i_puzzle:$i_puzzle | i_episode:$i_episode | path_length:$path_length")
            end
            reward = 0. # we have a single reward at goal state. If the episode was
            # not completed, this code still correctly works (updates are 0)
            for t in 1:path_length
                action_counter += 1
                state = episode_data[t, Int(c_state)]
                action = episode_data[t, Int(c_action)]
                reward = episode_data[t, Int(c_reward)]
                next_state = episode_data[t, Int(c_nextState)]
                done = reward>0 ? true : false

                p_actions = getSoftmaxProbs([learner.θ[state,1],learner.θ[state,2]], temp, [0., 0.])
                LL_unbiased += log(p_actions[action])
                probs_unbiased[action_counter] = copy(p_actions[action])

                PolicyParametersImbalance_record[action_counter] = abs(learner.θ[state, 1] - learner.θ[state, 2])
                PolicyImbalance_record[action_counter] = abs(p_actions[1] - p_actions[2])
                PolicyParametersImbalanceAllstates_record[action_counter] = abs.(learner.θ[:, 1] - learner.θ[:, 2])
                PolicyParametersDiffChosenVsUnchosen_record[action_counter] = learner.θ[state, action] - learner.θ[state, mod(action,2)+1]
                PolicyDiffChosenVsUnchosen_record[action_counter] = p_actions[action] - p_actions[mod(action,2)+1]

                θ_before = deepcopy(learner.θ)

                update!(learner, state, action, reward, p_actions[action], modulatingfactor)
                Δpolicy_record[action_counter] = sum(abs.(learner.θ .- θ_before))

                if do_debug_print
                    println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | p(s,a):$(round(p_action,digits=2))")
                end
            end # path_length (in single episode)
        end # episode
    end # puzzzle
  #---- Without bias
  LL = LL_unbiased
  trial_by_trial_p_sa = (probs_unbiased)

  algo = "REINFORCE"
  param_v = [alpha, gamma, temp]
  param_str = "alpha=$(round(alpha,digits=4)), gamma=$(round(gamma,digits=4)), temp=$(round(temp,digits=4))"

  results_dict = Dict{String,Any}("final_Theta" => learner.θ,
                                "DeltaPolicy" => Δpolicy_record,
                                "PolicyParametersImbalance" => PolicyParametersImbalance_record,
                                "PolicyImbalance" => PolicyImbalance_record,
                                "PolicyParametersImbalanceAllstates" => PolicyParametersImbalanceAllstates_record,
                                "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record,
                                "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record)

  signals = Step_by_step_signals(
    data_stats.all_subject_ids[1],
    [], # no RPE
    [], # no SPE
    trial_by_trial_p_sa,
    LL,
    data_stats,
    algo,
    param_v,
    param_str,
    99, #biased_action_idx, dummy value
    results_dict)
  return LL, signals

end
