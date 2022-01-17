using DataStructures

"""
Hybrid Actor-critic
-------------------

Hybrid model : Actor-critic & Forward learner

See learner/forwardlearner.jl, learner/actorcriticlearner.jl
and learner/hybridlearner.jl for the core implementation.

Note 1
------
get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl

Note 2
------
Temperature of softmax is removed, because this learner in NOT convex and weights
can be in (0, +Inf). The temperature is absorbed in the weights. Otherwise
it is overparametrized.
"""
function getLL_Hybrid_AC_FWD(data::Array{Int,2},
                            alpha::Float64, eta_critic::Float64,
                            gamma::Float64, lambda_actor::Float64,
                            lambda_critic::Float64,
                            eta_fwd::Float64,
                            w_MB::Float64,
                            w_MF::Float64;
                            init_R = 0.,
                            init_policy=nothing,
                            init_Vs = nothing,
                            use_replacing_trace=true,
                            do_debug_print = false)

    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_Hybrid_AC_FWD supports only one subject")
    end
    all_puzzles = get_all_puzzles(data)

    LL = 0.
    action_selection_probs = zeros(Float64, nr_samples)
    # signals:
    TD_record = zeros(Float64, nr_samples)
    SPE_record = zeros(Float64, nr_samples)
    V_state = zeros(Float64, nr_samples + data_stats.nr_rewards)

    PolicyParametersDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)
    PolicyDiffChosenVsUnchosen_record = zeros(Float64, nr_samples)

    learnerMB = []
    learnerMF = []
    learnerHybrid = []
    modulatingfactor = 0.
    action_counter = 0
    for i_puzzle in all_puzzles
        # Puzzle = independent block of the experiment.
        # The code was made for general purposes, in this experiment we
        # have only ONE puzzle.
        learnerMB = ForwardLearner(nr_states = nr_states,
                            nr_actions = nr_actions,
                            η = eta_fwd, γ = gamma,
                            init_R = 0.
                            )
        if init_policy == nothing # providing init_policy w/o init_Vs is considered a bug
            learnerMF = ActorCritic(nr_states = nr_states,
                                nr_actions = nr_actions,
                                α_critic = eta_critic,
                                α_actor = alpha,
                                γ = gamma,
                                λ_critic = lambda_critic,
                                λ_actor = lambda_actor,
                                use_replacing_trace = use_replacing_trace,
                                do_debug_print = do_debug_print)
        else
            learnerMF = ActorCritic(nr_states = nr_states,
                                nr_actions = nr_actions,
                                α_critic = eta_critic,
                                α_actor = alpha,
                                γ = gamma,
                                λ_critic = lambda_critic,
                                λ_actor = lambda_actor,
                                policyparameters = copy(init_policy[current_puzzle_id]),
                                V = copy(init_Vs[current_puzzle_id]),
                                use_replacing_trace = use_replacing_trace,
                                do_debug_print = do_debug_print)
        end
        learnerHybrid = Hybrid(nr_states = nr_states,
                            nr_actions = nr_actions,
                            w_MB = w_MB,
                            w_MF = w_MF
                            )
        all_epi_in_puzzle = get_all_episodes_in_puzzle(data, i_puzzle)
        j = 1
        for i_episode in all_epi_in_puzzle
            episode_data = get_data_by_puzzle_episode(data, i_puzzle, i_episode)
            path_length = size(episode_data)[1]  # = T
            if do_debug_print
                println("i_puzzle:$i_puzzle | i_episode:$i_episode | path_length:$path_length")
            end
            isendofexperiment = false

            for t in 1:path_length
                action_counter += 1
                state = episode_data[t, Int(c_state)]
                action = episode_data[t, Int(c_action)]
                reward = episode_data[t, Int(c_reward)]
                next_state = episode_data[t, Int(c_nextState)]
                done = reward>0 ? true : false
                # @show state, action, reward, next_state
                if do_debug_print
                    # println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | p(s,a):$(round(p_action,2)) | p(s,1;+-bias): $(round(p1_pos,2))/$(round(p1_neg,2))")
                    println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                end

                # Hybrid learner
                update!(learnerHybrid, state, action, reward, learnerMB.QSA, learnerMF.policyparameters) #Q_hybrid =  w_MB*learnerMB.QSA + w_MF*learnerMF.QSA

                p_actions = getSoftmaxProbs(learnerHybrid.QSA[state, :], 1., [0., 0.])
                action_selection_probs[action_counter] = copy(p_actions[action])
                LL += log(p_actions[action])

                PolicyParametersDiffChosenVsUnchosen_record[action_counter] = learnerHybrid.QSA[state, action] - learnerHybrid.QSA[state, mod(action,2)+1]
                PolicyDiffChosenVsUnchosen_record[action_counter] = p_actions[action] - p_actions[mod(action,2)+1]

                # ForwardLearner
                update!(learnerMB, state, action, next_state, reward)
                SPE_record[action_counter] = copy(learnerMB.spe[1])

                # Track V values (used as fmri regressor)
                V_state[j] = learnerMF.V[state]
                if done
                    j+=1
                    if j < (nr_samples + data_stats.nr_rewards)
                        V_state[j] = learnerMF.V[next_state] # Track V value of landing state in case it is the Goal
                    end
                end
                j+=1

                # Actor-Critic
                update!(learnerMF, state, action, next_state, reward, p_actions[action], modulatingfactor)
                TD_record[action_counter] = learnerMF.td_error[1]
            end # end t
        end # end i_episode
    end # end i_puzzle
    #---- Without bias
    trial_by_trial_p_sa = (action_selection_probs)
    if do_debug_print
        println(learnerHybrid.QSA)
        println("action_selection_probs: $action_selection_probs")
        println("LL_max: $LL")
        println("trial_by_trial_p_sa: $trial_by_trial_p_sa")
    end
    algo = "hybrid_actorcritic_FWD"

    param_v = [alpha, eta_critic, gamma, lambda_actor, lambda_critic, eta_fwd, w_MB, w_MF]
    param_str = "alpha=$(round(alpha,digits=4)),
                eta_critic=$(round(eta_critic,digits=4)),
                gamma=$(round(gamma,digits=4)),
                lambda_actor=$(round(lambda_actor,digits=4)),
                lambda_critic=$(round(lambda_critic,digits=4)),
                eta_fwd=$(round(eta_fwd,digits=4)),
                w_MB=$(round(w_MB,digits=4)),
                w_MF=$(round(w_MF,digits=4))"

    results_dict = Dict{String,Any}("finalQSA_Hybrid_record" => learnerHybrid.QSA,
                                    "final_PolicyParameters_MF" => learnerMF.policyparameters,
                                    "finalQSA_MB_record" => learnerMB.QSA,
                                    "final_Tsas" => learnerMB.TSAS,
                                    "trialByTrial_Vs" => V_state,
                                    "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record,
                                    "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record,
                                )
    signals = Step_by_step_signals(data_stats.all_subject_ids[1],
    TD_record, #RPE
    SPE_record, #SPE
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
