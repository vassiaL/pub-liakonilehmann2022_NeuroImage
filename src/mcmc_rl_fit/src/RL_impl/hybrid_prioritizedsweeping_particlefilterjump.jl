using DataStructures

"""
Hybrid-λ-PS-PF
--------------

Hybrid model : Prioritized Sweeping (with Particle Filtering) + SARSA-lambda
Prioritized Sweeping with: ParticleFilterJump for the Tsas' estimation
and Perfect Integration for Rsas' estimation.

See learner/prioritizedsweepinglearner.jl, learner/sarsaLlearner.jl and
learner/hybridlearner.jl for the core implementation.

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
function getLL_Hybrid_Prio_Sweep_TParticleFilterJump(data::Array{Int,2},
                                                gamma::Float64,
                                                surpriseprobability::Float64,
                                                stochasticity::Float64,
                                                nr_upd_cycles::Union{Int, Float64},
                                                nparticles::Union{Int, Float64},
                                                alpha::Float64,
                                                lambda::Float64,
                                                w_MB::Float64,
                                                w_MF::Float64;
                                                particlefilterseeds = [1,2,3],
                                                init_R = 0.,
                                                do_debug_print = false
                                                )
    nr_upd_cycles = converttoInt(nr_upd_cycles)
    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_Hybrid_Prio_Sweep_TParticleFilterJump supports only one subject")
    end
    all_puzzles = get_all_puzzles(data)

    action_selection_probs_seeds = Array{Array{Float64, 1}, 1}(undef, length(particlefilterseeds))
    [action_selection_probs_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    LL_seeds = zeros(Float64, length(particlefilterseeds))

    Tsas_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    # signals:
    RPE_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [RPE_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    Surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [Surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    γ_surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [γ_surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    QSA_Hybrid_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    QSA_MB_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    QSA_MF_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    learnerMB = []
    learnerMF = []
    learnerHybrid = []
    modulatingfactor = 0.
    for i_seed in 1:length(particlefilterseeds)
        action_counter = 0
        for i_puzzle in all_puzzles
            # Puzzle = independent block of the experiment.
            # The code was made for general purposes, in this experiment we
            # have only ONE puzzle.
            learnerMB = PrioritizedSweeping(nr_states = nr_states,
                                        nr_actions = nr_actions,
                                        γ = gamma, init_R = init_R,
                                        nr_upd_cycles = nr_upd_cycles,
                                        Testimatetype = TParticleFilterJump,
                                        nparticles = nparticles,
                                        surpriseprobability = surpriseprobability,
                                        stochasticity = stochasticity,
                                        seedlearner = particlefilterseeds[i_seed],
                                        do_debug_print = do_debug_print
                                        )
            learnerMF = SarsaL(nr_states = nr_states,
                                nr_actions = nr_actions,
                                α = alpha,
                                γ = gamma,
                                λ = lambda,
                                initvalue = 0.
                                )
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
                        println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                    end

                    # Hybrid learner
                    update!(learnerHybrid, state, action, reward, learnerMB.QSA, learnerMF.QSA) #Q_hybrid =  w_MB*learnerMB.QSA + w_MF*learnerMF.QSA

                    p_actions = getSoftmaxProbs(learnerHybrid.QSA[state, :], 1., [0., 0.])
                    action_selection_probs_seeds[i_seed][action_counter] = copy(p_actions[action])
                    LL_seeds[i_seed] += log(p_actions[action])

                    # Prioritized Sweeping
                    update!(learnerMB, state, action, next_state, reward, done)
                    γ_surprise_seeds[i_seed][action_counter] = copy(learnerMB.Testimate.γ_surprise[1])
                    Surprise_seeds[i_seed][action_counter] = copy(learnerMB.Testimate.Sgm[1])
                    Tsas_seeds[i_seed]= deepcopy(learnerMB.Testimate.Ps1a0s0)

                    # Sort out if end of episode or end of experiment
                    next_action = 0
                    if reward==0 #non goal state
                        if nr_samples > action_counter
                            next_action = data[action_counter+1,Int(c_action)]
                        else
                            isendofexperiment = true
                        end
                    end

                    # Sarsa-Lambda with replacing traces
                    update!(learnerMF, state, action, next_state, next_action, reward,
                            isendofexperiment, modulatingfactor)
                    RPE_seeds[i_seed][action_counter] = copy(learnerMF.rpe[1]) # Even though it is the same across seeds, save it anyway...

                    QSA_MB_seeds[i_seed] = copy(learnerMB.QSA)
                    QSA_MF_seeds[i_seed] = copy(learnerMF.QSA)
                    QSA_Hybrid_seeds[i_seed] = copy(learnerHybrid.QSA)

                end # end t
            end # end i_episode
        end # end i_puzzle
    end # end i_seed

    LL = get_meanLL_acrossseeds(LL_seeds)

    action_selection_probs_cat = hcat(action_selection_probs_seeds...)
    action_selection_probs = mean(action_selection_probs_cat, dims=2)[:]
    trial_by_trial_p_sa = (action_selection_probs)

    # For signals: return the ones corresponding to the Max LL across seeds
    LL_max, seed_idx = findmax(LL_seeds)
    RPE_record = copy(RPE_seeds[seed_idx])
    trial_by_trial_p_sa_max = (action_selection_probs_seeds[seed_idx])

    if do_debug_print
        println(QSA_Hybrid_seeds[seed_idx])
        println("action_selection_probs: $action_selection_probs")
        println("LL_mean: $LL_mean")
        println("LL_max: $LL_max")
        println("trial_by_trial_p_sa: $trial_by_trial_p_sa")
    end

    algo = "hybrid_priosweepparticlefilter_sarsaL"
    param_v = [nr_upd_cycles, nparticles, gamma, surpriseprobability,
                stochasticity, alpha, lambda, w_MB, w_MF]
    param_str = "nr_upd_cycles=$nr_upd_cycles, nparticles=$nparticles,
                gamma=$(round(gamma,digits=4)),
                surpriseprobability=$(round(surpriseprobability,digits=4)),
                stochasticity=$(round(stochasticity,digits=4)),
                alpha=$(round(alpha,digits=4)),
                lambda=$(round(lambda,digits=4)),
                w_MB=$(round(w_MB,digits=4)),
                w_MF=$(round(w_MF,digits=4))"

    results_dict = Dict{String,Any}("finalQSA_Hybrid_record" => QSA_Hybrid_seeds[seed_idx],
                                    "finalQSA_MF_record" => QSA_MF_seeds[seed_idx],
                                    "finalQSA_MB_record" => QSA_MB_seeds[seed_idx],
                                    "final_Tsas" => Tsas_seeds[seed_idx],
                                    "Surprise" => Surprise_seeds[seed_idx],
                                    "gamma_surprise" => γ_surprise_seeds[seed_idx],
                                    "nr_upd_cycles" => nr_upd_cycles,
                                    "nparticles" => nparticles,
                                    "particlefilterseeds" => particlefilterseeds,
                                    "trial_by_trial_p_sa_max" => trial_by_trial_p_sa_max)

    signals = Step_by_step_signals(data_stats.all_subject_ids[1],
    RPE_record, #RPE
    [], #SPE
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
