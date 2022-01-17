"""
SARSA-λ
-------

See learner/sarsaLlearner.jl for the core implementation.

get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl
"""
function getLL_SarsaLambda(data::Array{Int,2}, alpha::Float64, gamma::Float64,
                            lambda::Float64, temp::Float64;
                            reward_magnitude=1., action_cost = 0., init_Q=0.,
                            initQSA=nothing, use_replacing_trace=true)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_SarsaLambda supports only one subject")
    end

    LL_unbiased = 0.

    action_selection_probs = zeros(Float64, nr_samples)

    # signals:
    RPE_record = zeros(Float64, nr_samples)
    Qsa_record = zeros(Float64, nr_samples) # trial-by-trial action value
    Vs_record = zeros(Float64, nr_samples)  # trial-by-trial max(Q(s,:))
    finalQSA_record = Dict{Int,Any}() # end of Episode, Q(sa) estimate
    RPE_QV_record = zeros(Float64, nr_samples) # the RPE defined as  Q(s_t, a_t)  -  gamma*V(s_t+1), where V(s_t+1) = max(Q(s_t+1, :))

    # Record various values (not used eventually in paper)
    state_on_counter_V1 = 1; state_on_counter_V2 = 1; state_on_counter_V3 = 1
    total_nr_of_state_on = nr_samples + data_stats.nr_episodes  # actions + nr_goal states (no action at goal)
    StateValue_V1_record = zeros(Float64, total_nr_of_state_on)  # value = max(Q(s,:)), 0 at goal state
    StateValue_V2_record = zeros(Float64, total_nr_of_state_on)  # r + gamma*V_next
    StateValue_V3_record = zeros(Float64, total_nr_of_state_on)  # V AFTER update. Start state: V before update

    PolicyParametersImbalance_record = zeros(Float64, nr_samples)
    PolicyImbalance_record = zeros(Float64, nr_samples)
    PolicyParametersImbalanceAllstates_record = Array{Array{Float64,1}, 1}(undef, nr_samples)
    [PolicyParametersImbalanceAllstates_record[i] = zeros(Float64, nr_states) for i in 1:nr_samples]
    PolicyParametersDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)
    PolicyDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)

    current_puzzle_id = -1
    # Puzzle = independent block of the experiment.
    # The code was made for general purposes, in this experiment we
    # have only ONE puzzle.
    is_start_state = true
    isendofexperiment = false
    modulatingfactor = 0.
    learner = []
    for i = 1:nr_samples
        if current_puzzle_id != data[i,Int(c_puzzle)]
            current_puzzle_id = data[i,Int(c_puzzle)]
            # if new puzzle -> reset learning
            if initQSA == nothing
                learner = SarsaL(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α = alpha, γ = gamma,
                                    λ = lambda, initvalue = init_Q,
                                    use_replacing_trace = use_replacing_trace
                                    )
            else
                learner = SarsaL(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α = alpha,
                                    γ = gamma,
                                    λ = lambda,
                                    QSA = copy(initQSA[current_puzzle_id]),
                                    use_replacing_trace = use_replacing_trace
                                    )
            end
        end
        state = data[i, Int(c_state)]
        action = data[i, Int(c_action)]
        reward::Float64= data[i, Int(c_reward)]
        reward *= reward_magnitude
        next_state = data[i, Int(c_nextState)]

        Qsa_record[i] = copy(learner.QSA[state, action])
        Vs_record[i] = copy(maximum(learner.QSA[state, :]))

        StateValue_V1_record[state_on_counter_V1] = maximum(learner.QSA[state, :])
        state_on_counter_V1 += 1
        StateValue_V2_record[state_on_counter_V2] = learner.γ*maximum(learner.QSA[state, :])
        state_on_counter_V2 += 1

        # and if goal state:
        if reward>0
            StateValue_V1_record[state_on_counter_V1]  = 0
            state_on_counter_V1 += 1
            StateValue_V2_record[state_on_counter_V2] = reward #+ action_cost
            state_on_counter_V2 += 1
        end

        if is_start_state
            StateValue_V3_record[state_on_counter_V3] = maximum(learner.QSA[state, :])
            state_on_counter_V3 += 1
        end

        p_actions = getSoftmaxProbs(learner.QSA[state,:], temp, [0., 0.])
        action_selection_probs[i] = copy(p_actions[action])
        LL_unbiased += log(p_actions[action])

        PolicyParametersImbalance_record[i] = abs(learner.QSA[state, 1] - learner.QSA[state, 2])
        PolicyImbalance_record[i] = abs(p_actions[1] - p_actions[2])
        PolicyParametersImbalanceAllstates_record[i] = abs.(learner.QSA[:, 1] - learner.QSA[:, 2])
        PolicyParametersDiffChosenVsUnchosen_record[i] = learner.QSA[state, action] - learner.QSA[state, mod(action,2)+1]
        PolicyDiffChosenVsUnchosen_record[i] = p_actions[action] - p_actions[mod(action,2)+1]

        # Sort out if end of episode or end of experiment
        next_action = 0
        if reward==0 #non goal state
            rpe_qv = learner.γ * maximum(learner.QSA[next_state, :]) - learner.QSA[state, action]
            if nr_samples > i
                next_action = data[i+1,Int(c_action)]
            else
                # println("i:$i \nS:$state | a:$action | R:$reward | S':$next_state")
                isendofexperiment = true
            end
        end

        update!(learner, state, action, next_state, next_action, reward,
                isendofexperiment, modulatingfactor)

        if reward>0; rpe_qv = copy(learner.rpe[1]); end

        RPE_record[i] = copy(learner.rpe[1])
        RPE_QV_record[i] = copy(rpe_qv)

        finalQSA_record[current_puzzle_id] = copy(learner.QSA)

        StateValue_V3_record[state_on_counter_V3] = maximum(learner.QSA[state, :])
        state_on_counter_V3 += 1
        is_start_state = reward>0 # next state will be a start state
    end

    maxLL = LL_unbiased
    trial_by_trial_p_sa = (action_selection_probs)

    trace_type = use_replacing_trace ? "ReplTrc" : "EligTrc"
    algo = "Sarsa-Lambda ($trace_type)"

    param_v = [alpha, gamma, lambda, temp,reward_magnitude, action_cost, init_Q]
    param_str = "alpha=$(round(alpha,digits=4)), gamma=$(round(gamma,digits=4)),
                lambda=$(round(lambda,digits=4)), temp=$(round(temp,digits=4)),
                reward_magnitude=$(round(reward_magnitude,digits=4)),
                action_cost=$(round(action_cost,digits=4)),
                initQSA=$(repr(initQSA)), use_replacing_trace=$use_replacing_trace"

    results_dict = Dict{String,Any}("trialByTrial_Qsa" => Qsa_record,
                                    "final_QSA" => finalQSA_record,
                                    "trialByTrial_Vs" => Vs_record,
                                    "RPE_QV" => RPE_QV_record,
                                    "StateValue_V1_record" => StateValue_V1_record,
                                    "StateValue_V2_record" => StateValue_V2_record,
                                    "StateValue_V3_record" => StateValue_V3_record,
                                    "PolicyParametersImbalance" => PolicyParametersImbalance_record,
                                    "PolicyImbalance" => PolicyImbalance_record,
                                    "PolicyParametersImbalanceAllstates" => PolicyParametersImbalanceAllstates_record,
                                    "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record,
                                    "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record)

    signals = Step_by_step_signals(data_stats.all_subject_ids[1],
    RPE_record, #RPE
    [], #SPE
    trial_by_trial_p_sa,
    maxLL,
    data_stats,
    algo,
    param_v,
    param_str,
    99, #biased_action_idx, dummy value
    results_dict)
    return maxLL, signals
end
