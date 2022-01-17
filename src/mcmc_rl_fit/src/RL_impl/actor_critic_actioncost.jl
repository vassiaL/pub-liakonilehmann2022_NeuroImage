"""
Actor-critic-cost
-----------------

Actor-Critic with an extra free parameter as an action cost,
that encourages the agent to find the most efficient path to the goal.
(See Appendix A.17)

See actor-critic.jl for some more implementational info.
See learner/actorcriticlearner.jl for the core implementation.

Note
----
get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl

"""
function getLL_ActorCritic_TDLambda_actioncost(data::Array{Int,2}, alpha::Float64, eta::Float64,
                                    gamma::Float64, lambda_actor::Float64,
                                    lambda_critic::Float64,
                                    temp::Float64,
                                    actioncost::Float64;
                                    init_policy=nothing,
                                    init_Vs = nothing,
                                    use_replacing_trace=true,
                                    do_debug_print = false)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_ActorCritic_TDLambda_actioncost supports only one subject")
    end

    LL_unbiased = 0.
    action_selection_probs = zeros(Float64, nr_samples)
    # signals:
    TD_record = zeros(Float64, nr_samples)
    V_state = zeros(Float64, nr_samples) # V_state = zeros(Float64, nr_samples + data_stats.nr_rewards)
    finalV_record = Dict{Int,Any}()
    final_policy_record = Dict{Int,Any}()
    Δpolicy_record = zeros(Float64, nr_samples)

    PolicyParametersImbalance_record = zeros(Float64, nr_samples)
    PolicyImbalance_record = zeros(Float64, nr_samples)
    PolicyParametersImbalanceAllstates_record = Array{Array{Float64,1}, 1}(undef, nr_samples)
    [PolicyParametersImbalanceAllstates_record[i] = zeros(Float64, nr_states) for i in 1:nr_samples]
    PolicyParametersDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)
    PolicyDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)

    current_puzzle_id = -1
    modulatingfactor = 1.
    learner = []
    for i = 1:nr_samples
        if current_puzzle_id != data[i,Int(c_puzzle)]
            # Puzzle = independent block of the experiment.
            # The code was made for general purposes, in this experiment we
            # have only ONE puzzle.
            current_puzzle_id = data[i,Int(c_puzzle)]
            if do_debug_print
              println("new puzzle: $current_puzzle_id")
            end
            if init_policy == nothing # providing init_policy w/o init_Vs is considered a bug
                learner = ActorCritic(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α_critic = eta,
                                    α_actor = alpha,
                                    γ = gamma,
                                    λ_critic = lambda_critic,
                                    λ_actor = lambda_actor,
                                    use_replacing_trace = use_replacing_trace,
                                    do_debug_print = do_debug_print)
            else
                learner = ActorCritic(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α_critic = eta,
                                    α_actor = alpha,
                                    γ = gamma,
                                    λ_critic = lambda_critic,
                                    λ_actor = lambda_actor,
                                    policyparameters = copy(init_policy[current_puzzle_id]),
                                    V = copy(init_Vs[current_puzzle_id]),
                                    use_replacing_trace = use_replacing_trace,
                                    do_debug_print = do_debug_print,
                                    actioncost = actioncost)
            end
        end
        state = data[i, Int(c_state)]
        action = data[i, Int(c_action)]
        reward::Float64= data[i, Int(c_reward)]
        next_state = data[i, Int(c_nextState)]
        # @show state, action, reward, next_state

        p_actions = getSoftmaxProbs(learner.policyparameters[state,:], temp, [0., 0.])

        unbiased_p_sa = p_actions[action]
        action_selection_probs[i] = copy(unbiased_p_sa)
        LL_unbiased += log(unbiased_p_sa)

        V_state[i] = learner.V[state]
        PolicyParametersImbalance_record[i] = abs(learner.policyparameters[state, 1] - learner.policyparameters[state, 2])
        PolicyImbalance_record[i] = abs(p_actions[1] - p_actions[2])
        PolicyParametersImbalanceAllstates_record[i] = abs.(learner.policyparameters[:, 1] - learner.policyparameters[:, 2])
        PolicyParametersDiffChosenVsUnchosen_record[i] = learner.policyparameters[state, action] - learner.policyparameters[state, mod(action,2)+1]
        PolicyDiffChosenVsUnchosen_record[i] = p_actions[action] - p_actions[mod(action,2)+1]

        policyparameters_before = deepcopy(learner.policyparameters)

        # Actor-Critic
        update!(learner, state, action, next_state, reward, unbiased_p_sa, modulatingfactor)
        TD_record[i] = learner.td_error[1] # Track td_error AFTER update (right?)

        finalV_record[current_puzzle_id] = learner.V
        final_policy_record[current_puzzle_id] = learner.policyparameters
        Δpolicy_record[i] = sum(abs.(learner.policyparameters .- policyparameters_before))
    end

    maxLL = LL_unbiased
    trial_by_trial_p_sa = (action_selection_probs)

    trace_type = use_replacing_trace ? "ReplTrc" : "EligTrc"
    algo = "Actor-Critic (TD-Lambda, $trace_type)"
    param_v = [alpha, eta, gamma, lambda_actor, lambda_critic, temp]
    param_str = "alpha(actor)=$(round(alpha,digits=4)), eta(critic)=$(round(eta,digits=4)),
                gamma=$(round(gamma,digits=4)),
                lambda_actor=$(round(lambda_actor,digits=4)),
                lambda_critic=$(round(lambda_critic,digits=4)),
                temp=$(round(temp,digits=4)),
                init_policy=$(repr(init_policy)),
                init_Vs=$(repr(init_Vs)), use_replacing_trace=$use_replacing_trace"

    results_dict = Dict{String,Any}(
        "trialByTrial_Vs" => V_state,
        "final_V" => finalV_record,
        "final_Policy" => final_policy_record,
        "DeltaPolicy" => Δpolicy_record,
        "PolicyParametersImbalance" => PolicyParametersImbalance_record,
        "PolicyImbalance" => PolicyImbalance_record,
        "PolicyParametersImbalanceAllstates" => PolicyParametersImbalanceAllstates_record,
        "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record,
        "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record,
        )

    signals = Step_by_step_signals(
    data_stats.all_subject_ids[1],
    TD_record,  # RPE
    [],  # no SPE
    trial_by_trial_p_sa,
    maxLL,
    data_stats,
    algo,
    param_v,
    param_str,
    99, #dummy value
    results_dict)
    return maxLL, signals
end
