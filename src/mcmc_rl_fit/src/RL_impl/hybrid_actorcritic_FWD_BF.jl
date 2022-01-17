using DataStructures

"""
Hybrid Actor-critic PF
----------------------

Hybrid model : Actor-critic & Value iteration with Particle filter

This algorithm is a weighted sum of Actor-critic and a model-based algorithm
that implements value iteration and estimates the
transition matrix via a particle filter.
In other words, the model-based structure is the same as in
Forward Learner, but now the world-model is estimated optimally and not via a
delta rule.

See learner/forwardlearnerBF.jl, learner/actorcriticlearner.jl
and learner/hybridlearner.jl for the core implementation.

Note 1
------
With particle filtering we need to run multiple times with different seeds.
As final LL we report the MEAN across seeds.
For signals/results we keep track of the ones of all seeds
and in the end report the one with the MAXIMUM LL.

Note 2
------
get_all_puzzles(), get_all_episodes_in_puzzle(), get_data_by_puzzle_episode()
are in rl_data_toolsforalgos.jl

Note 3
------
Temperature of softmax is removed, because this learner in NOT convex and weights
can be in (0, +Inf). The temperature is absorbed in the weights. Otherwise we
it is overparametrized.
"""
function getLL_Hybrid_AC_FWD_BF(data::Array{Int,2},
                            alpha::Float64, eta_critic::Float64,
                            gamma::Float64, lambda_actor::Float64,
                            lambda_critic::Float64,
                            surpriseprobability::Float64,
                            stochasticity::Float64,
                            nparticles::Union{Int, Float64},
                            w_MB::Float64,
                            w_MF::Float64;
                            particlefilterseeds = [1,2,3],
                            init_R = 0.,
                            init_policy=nothing,
                            init_Vs = nothing,
                            use_replacing_trace=true,
                            do_debug_print = false)

    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_Hybrid_AC_FWD supports only one subject")
    end
    all_puzzles = get_all_puzzles(data)

    action_selection_probs_seeds = Array{Array{Float64, 1}, 1}(undef, length(particlefilterseeds))
    [action_selection_probs_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    LL_seeds = zeros(Float64, length(particlefilterseeds))

    Tsas_seeds = Array{Any, 1}(undef, length(particlefilterseeds))

    # signals:
    TD_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [TD_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    Surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [Surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    V_state_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [V_state_seeds[i_seed] = zeros(Float64, nr_samples + data_stats.nr_rewards) for i_seed in 1:length(particlefilterseeds)]
    γ_surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [γ_surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    SPE_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [SPE_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    final_policyparameters_record_seed = Array{Dict{Int,Any}, 1}(undef, length(particlefilterseeds))
    [final_policyparameters_record_seed[i_seed] = Dict{Int,Any}() for i_seed in 1:length(particlefilterseeds)]
    finalQSA_Hybrid_record_seed = Array{Dict{Int,Any}, 1}(undef, length(particlefilterseeds))
    [finalQSA_Hybrid_record_seed[i_seed] = Dict{Int,Any}() for i_seed in 1:length(particlefilterseeds)]
    finalQSA_MB_record_seed = Array{Dict{Int,Any}, 1}(undef, length(particlefilterseeds))
    [finalQSA_MB_record_seed[i_seed] = Dict{Int,Any}() for i_seed in 1:length(particlefilterseeds)]

    for i_seed in 1:length(particlefilterseeds)
        learnerMB = []
        learnerMF = []
        learnerHybrid = []
        modulatingfactor = 0.
        action_counter = 0
        for i_puzzle in all_puzzles
            # Puzzle = independent block of the experiment.
            # The code was made for general purposes, in this experiment we
            # have only ONE puzzle.
            learnerMB = ForwardLearnerBF(nr_states = nr_states,
                            nr_actions = nr_actions,
                            γ = gamma,
                            init_R = 0.,
                            nparticles = nparticles,
                            surpriseprobability = surpriseprobability,
                            stochasticity = stochasticity,
                            seedlearner = particlefilterseeds[i_seed],
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
                    if do_debug_print
                        # println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | p(s,a):$(round(p_action,2)) | p(s,1;+-bias): $(round(p1_pos,2))/$(round(p1_neg,2))")
                        println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                    end

                    # Hybrid learner
                    update!(learnerHybrid, state, action, reward, learnerMB.QSA, learnerMF.policyparameters) #Q_hybrid =  w_MB*learnerMB.QSA + w_MF*learnerMF.QSA

                    p_actions = getSoftmaxProbs(learnerHybrid.QSA[state, :], 1., [0., 0.])

                    action_selection_probs_seeds[i_seed][action_counter] = copy(p_actions[action])
                    LL_seeds[i_seed] += log(p_actions[action])

                    # ForwardLearner-like update (ie Value iteration),
                    # but with model estimated via particle filter
                    update!(learnerMB, state, action, next_state, reward)
                    γ_surprise_seeds[i_seed][action_counter] = copy(learnerMB.Testimate.γ_surprise[1])
                    Surprise_seeds[i_seed][action_counter] = copy(learnerMB.Testimate.Sgm[1])
                    SPE_seeds[i_seed][action_counter] = copy(learnerMB.Testimate.SPE[1])

                    # Actor-Critic
                    update!(learnerMF, state, action, next_state, reward, p_actions[action], modulatingfactor)
                    TD_seeds[i_seed][action_counter] = copy(learnerMF.td_error[1])

                end # end t
            end # end i_episode
            final_policyparameters_record_seed[i_seed][i_puzzle] = copy(learnerMF.policyparameters)
            finalQSA_Hybrid_record_seed[i_seed][i_puzzle] = copy(learnerHybrid.QSA)
            finalQSA_MB_record_seed[i_seed][i_puzzle] = copy(learnerMB.QSA)
        end # end i_puzzle
        Tsas_seeds[i_seed]= deepcopy(learnerMB.Testimate.Ps1a0s0)
    end # seeds

    LL = get_meanLL_acrossseeds(LL_seeds)

    action_selection_probs_cat = hcat(action_selection_probs_seeds...)
    action_selection_probs = mean(action_selection_probs_cat, dims=2)[:]
    trial_by_trial_p_sa = (action_selection_probs)

    # For signals: return the ones corresponding to the Max LL across seeds
    LL_max, seed_idx = findmax(LL_seeds)
    TD_record = copy(TD_seeds[seed_idx])
    SPE_record = copy(SPE_seeds[seed_idx])
    trial_by_trial_p_sa_max = (action_selection_probs_seeds[seed_idx])

    algo = "hybrid_actorcritic_FWD_BF"

    param_v = [alpha, eta_critic, gamma, lambda_actor, lambda_critic, surpriseprobability, stochasticity, nparticles, w_MB, w_MF]
    param_str = "alpha=$(round(alpha,digits=4)),
                eta_critic=$(round(eta_critic,digits=4)),
                gamma=$(round(gamma,digits=4)),
                lambda_actor=$(round(lambda_actor,digits=4)),
                lambda_critic=$(round(lambda_critic,digits=4)),
                surpriseprobability=$(round(surpriseprobability,digits=4)),
                stochasticity=$(round(stochasticity,digits=4)),
                nparticles=$nparticles,
                w_MB=$(round(w_MB,digits=4)),
                w_MF=$(round(w_MF,digits=4))"

    results_dict = Dict{String,Any}("finalQSA_Hybrid_record" => finalQSA_Hybrid_record_seed[seed_idx],
                                    "final_PolicyParameters_MF" => final_policyparameters_record_seed[seed_idx],
                                    "finalQSA_MB_record" => finalQSA_MB_record_seed[seed_idx],
                                    "final_Tsas" => Tsas_seeds[seed_idx],
                                    "Surprise" => Surprise_seeds[seed_idx],
                                    "gamma_surprise" => γ_surprise_seeds[seed_idx],
                                    "nparticles" => nparticles,
                                    "particlefilterseeds" => particlefilterseeds,
                                    "trial_by_trial_p_sa_max" => trial_by_trial_p_sa_max)

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
