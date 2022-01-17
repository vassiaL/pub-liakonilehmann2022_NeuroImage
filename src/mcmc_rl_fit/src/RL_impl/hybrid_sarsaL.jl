"""
# Hybrid-λ (Daw et al. 2011)
# --------------------------

Hybrid: Forward Learner + SARSA-λ

See learner/forwardlearner.jl, learner/sarsaLlearner.jl
and learner/hybridconvexlearner.jl for the core implementation.

"""
function getLL_HybridSarsaLambda(data::Array{Int,2}, alpha::Float64, gamma::Float64,
                                lambda::Float64, eta::Float64, decay_offset::Float64,
                                decay_slope::Float64, temp::Float64)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_HybridSarsaLambda_biased supports only one subject")
    end

    LL_unbiased = 0.
    action_selection_probs = zeros(Float64, nr_samples)

    # signals:
    RPE_record = zeros(Float64, nr_samples)
    SPE_record = zeros(Float64, nr_samples)
    Qsa_hyb_record = zeros(Float64, nr_samples)
    learnerMB = []
    learnerMF = []
    learnerHybrid = []
    current_puzzle_id = -1
    # Puzzle = independent block of the experiment.
    # The code was made for general purposes, in this experiment we
    # have only ONE puzzle.
    isendofexperiment = false
    modulatingfactor = 0.
    for i = 1:nr_samples
        if current_puzzle_id != data[i,Int(c_puzzle)]
            # if new puzzle -> reset learning
            current_puzzle_id = data[i,Int(c_puzzle)]
            learnerMB = ForwardLearner(nr_states = nr_states,
                                nr_actions = nr_actions,
                                η = eta, γ = gamma,
                                init_R = 0.
                                )
            learnerMF = SarsaL(nr_states = nr_states,
                                nr_actions = nr_actions,
                                α = alpha,
                                γ = gamma,
                                λ = lambda,
                                initvalue = 0.
                                )
            learnerHybrid = HybridConvex(nr_states = nr_states,
                                nr_actions = nr_actions,
                                decay_offset = decay_offset,
                                decay_slope = decay_slope
                                )
        end
        state = data[i, Int(c_state)]
        action = data[i, Int(c_action)]
        reward = data[i, Int(c_reward)]
        next_state = data[i, Int(c_nextState)]

        # Hybrid learner
        update!(learnerHybrid, state, action, reward, learnerMB.QSA, learnerMF.QSA) #Q_hybrid =  w_MB*learnerMB.QSA + w_MF*learnerMF.QSA
        Qsa_hyb_record[i] = copy(learnerHybrid.QSA[state, action])

        p_actions = getSoftmaxProbs(learnerHybrid.QSA[state,:], temp, [0., 0.])
        action_selection_probs[i] = copy(p_actions[action])
        LL_unbiased += log(p_actions[action])

        # ForwardLearner
        update!(learnerMB, state, action, next_state, reward)
        SPE_record[i] = copy(learnerMB.spe[1])

        # Sort out if end of episode or end of experiment
        next_action = 0
        if reward==0 #non goal state
            if nr_samples > i
                next_action = data[i+1,Int(c_action)]
            else
                isendofexperiment = true
            end
        end

        # Sarsa-Lambda with replacing traces
        update!(learnerMF, state, action, next_state, next_action, reward,
                isendofexperiment, modulatingfactor)
        RPE_record[i] = copy(learnerMF.rpe[1])
    end

    LL = LL_unbiased
    trial_by_trial_p_sa = (action_selection_probs)
    algo = "Hybrid FWD+SarsaL"
    results_dict = Dict{String,Any}("Qsa_hyb_record" => Qsa_hyb_record,
                                    "finalQSA_Hybrid_record" => learnerHybrid.QSA,
                                    "finalQSA_MF_record" => learnerMF.QSA,
                                    "finalQSA_MB_record" => learnerMB.QSA,
                                    "final_Tsas" => learnerMB.TSAS)
    param_v = [alpha, lambda, gamma, eta, decay_offset, decay_slope, temp]
    parma_str = "alpha=$(round(alpha,digits=4)), lambda=$(round(lambda,digits=4)),
                gamma=$(round(gamma,digits=4)), eta=$(round(eta,digits=4)),
                decay_offset=$(round(decay_offset,digits=4)),
                decay_slope=$(round(decay_slope,digits=4)), temp=$(round(temp,digits=4))"

    signals = Step_by_step_signals(data_stats.all_subject_ids[1],
        RPE_record,
        SPE_record,
        trial_by_trial_p_sa,
        LL,
        data_stats,
        algo,
        param_v,
        parma_str,
        99, #biased_action_idx, dummy value
        results_dict)
        return LL, signals
end
