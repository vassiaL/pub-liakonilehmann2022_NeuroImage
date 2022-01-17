using DataStructures
using Statistics

"""
PS with PF
----------

Prioritized Sweeping with: ParticleFilterJump for the Tsas' estimation
and Perfect Integration for Rsas' estimation.

See learner/prioritizedsweepinglearner.jl for the core implementation and more info.

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
"""
function getLL_Prio_Sweep_TParticleFilterJump(data::Array{Int,2},
                                                gamma::Float64, temp::Float64,
                                                surpriseprobability::Float64,
                                                stochasticity::Float64,
                                                nr_upd_cycles::Union{Int, Float64},
                                                nparticles::Union{Int, Float64};
                                                particlefilterseeds = [1,2,3],
                                                init_R = 0.,
                                                do_debug_print = false
                                                )
    nr_upd_cycles = converttoInt(nr_upd_cycles)
    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_Prio_Sweep_TParticleFilter supports only one subject")
    end
    all_puzzles = get_all_puzzles(data)

    LL_seeds = zeros(Float64, length(particlefilterseeds))
    action_selection_probs_seeds = Array{Array{Float64, 1}, 1}(undef, length(particlefilterseeds))
    [action_selection_probs_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    # signals:
    Surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [Surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    γ_surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [γ_surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    QSA_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    Tsas_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    learner = []
    for i_seed in 1:length(particlefilterseeds)
        action_counter = 0
        for i_puzzle in all_puzzles
            # Puzzle = independent block of the experiment.
            # The code was made for general purposes, in this experiment we
            # have only ONE puzzle.
            learner = PrioritizedSweeping(nr_states = nr_states,
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
            all_epi_in_puzzle = get_all_episodes_in_puzzle(data, i_puzzle)
            for i_episode in all_epi_in_puzzle
                episode_data = get_data_by_puzzle_episode(data, i_puzzle, i_episode)
                path_length = size(episode_data)[1]  # = T
                if do_debug_print
                    println("i_puzzle:$i_puzzle | i_episode:$i_episode | path_length:$path_length")
                end
                for t in 1:path_length
                    action_counter += 1
                    state = episode_data[t, Int(c_state)]
                    action = episode_data[t, Int(c_action)]
                    reward = episode_data[t, Int(c_reward)]
                    next_state = episode_data[t, Int(c_nextState)]
                    done = reward>0 ? true : false
                    # println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                    if do_debug_print
                        println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                    end

                    p_actions = getSoftmaxProbs(learner.QSA[state,:], temp, [0., 0.])
                    action_selection_probs_seeds[i_seed][action_counter] = copy(p_actions[action])
                    LL_seeds[i_seed] += log(p_actions[action])
                    if do_debug_print
                        println("p(s,a)=$p_actions, action=$action")
                    end

                    # Prioritized Sweeping with Particle Filtering
                    update!(learner, state, action, next_state, reward, done)

                    Surprise_seeds[i_seed][action_counter] = copy(learner.Testimate.Sgm[1])
                    γ_surprise_seeds[i_seed][action_counter] = copy(learner.Testimate.γ_surprise[1])
                    QSA_seeds[i_seed] = copy(learner.QSA)
                    Tsas_seeds[i_seed]= deepcopy(learner.Testimate.Ps1a0s0)

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
    trial_by_trial_p_sa_max = (action_selection_probs_seeds[seed_idx])
    if do_debug_print
        println(finalQSA_record)
        println("action_selection_probs: $action_selection_probs")
        println("LL_mean: $LL_mean")
        println("LL_max: $LL_max")
        println("trial_by_trial_p_sa: $trial_by_trial_p_sa")
    end
    algo = "prio_sweep_particlefilter"
    param_v = [nr_upd_cycles, nparticles, gamma, temp, surpriseprobability, stochasticity]
    param_str = "nr_upd_cycles=$nr_upd_cycles, nparticles=$nparticles,
                gamma=$(round(gamma,digits=4)), temp=$(round(temp,digits=4)),
                surpriseprobability=$(round(surpriseprobability,digits=4)),
                stochasticity=$(round(stochasticity,digits=4))"
    results_dict = Dict{String,Any}("final_QSA" => QSA_seeds[seed_idx],
                                    "final_Tsas" => Tsas_seeds[seed_idx],
                                    "Surprise" => Surprise_seeds[seed_idx],
                                    "gamma_surprise" => γ_surprise_seeds[seed_idx],
                                    "nr_upd_cycles" => nr_upd_cycles,
                                    "nparticles" => nparticles,
                                    "particlefilterseeds" => particlefilterseeds,
                                    "trial_by_trial_p_sa_max" => trial_by_trial_p_sa_max)

    signals = Step_by_step_signals(data_stats.all_subject_ids[1],
    [], #RPE
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
