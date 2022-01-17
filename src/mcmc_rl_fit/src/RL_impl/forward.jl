"""

FWD (Glaescher et al. 2010)
---------------------------

Forward Learner

See learner/forwardlearner.jl for the core implementation.

"""
function getLL_Forward_Learner(data::Array{Int,2}, eta::Float64,
                                gamma::Float64, temp::Float64;
                                initTsas = nothing, initR = nothing)

    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_Forward_Learner supports only one subject")
    end

    LL_unbiased = 0.
    action_selection_probs = zeros(Float64, nr_samples)
    # signals:
    SPE_record = zeros(Float64, nr_samples)
    RiskPE_record = zeros(Float64, nr_samples)
    Qsa_fwd_record = zeros(Float64, nr_samples)

    final_TSAS_record = Dict{Int,Any}()
    final_Reward_record = Dict{Int,Any}()
    current_puzzle_id = -1
    # Puzzle = independent block of the experiment.
    # The code was made for general purposes, in this experiment we
    # have only ONE puzzle.
    learner = []
    for i = 1:nr_samples-1
        if current_puzzle_id != data[i,Int(c_puzzle)]
            current_puzzle_id = data[i,Int(c_puzzle)]
            # new puzzle. reset learning
            if initTsas == nothing
                learner = ForwardLearner(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    η = eta, γ = gamma,
                                    init_R = 0.,
                                    )
            else
                learner = ForwardLearner(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    η = eta, γ = gamma,
                                    TSAS = copy(initTsas[current_puzzle_id]), # first argument, i-th puzzle
                                    R = copy(initR[current_puzzle_id])
                                    )
            end
        end
        state = data[i, Int(c_state)]
        action = data[i, Int(c_action)]
        reward = data[i, Int(c_reward)]
        next_state = data[i, Int(c_nextState)]
        Qsa_fwd_record[i] = copy(learner.QSA[state,action])

        p_actions = getSoftmaxProbs(learner.QSA[state,:], temp, [0., 0.])
        action_selection_probs[i] = copy(p_actions[action])
        LL_unbiased += log(p_actions[action])

        # ForwardLearner
        update!(learner, state, action, next_state, reward)
        SPE_record[i] = copy(learner.spe[1])

        final_TSAS_record[current_puzzle_id] = copy(learner.TSAS)
        final_Reward_record[current_puzzle_id] = copy(learner.R)
    end

    results_dict = Dict{String,Any}(
        "QSA_record"=> Qsa_fwd_record,
        "final_Tsas" => final_TSAS_record,
        "final_Qsa" => learner.QSA,
        "final_Reward" => final_Reward_record
        )

    LL = LL_unbiased
    trial_by_trial_p_sa = (action_selection_probs)
    algo = "Forward Learner"
    param_v = [eta, gamma, temp]
    parma_str = "eta=$(round(eta,digits=4)), gamma=$(round(gamma,digits=4)),
                temp=$(round(temp,digits=4)),
                initTsas=$(repr(initTsas)), initR=$(repr(initR))"
    signals = Step_by_step_signals(
        data_stats.all_subject_ids[1],
        [], #RPE
        SPE_record,
        trial_by_trial_p_sa,
        LL,
        data_stats,
        algo,
        param_v,
        parma_str,
        99, #biased_action_idx, dummy value
        results_dict
        )
    return LL, signals
end
