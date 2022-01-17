using DataStructures

"""
Surprise SARSA-λ
----------------

SARSA-lambda with surprise-modulated learning rate
with Particle Filtering

See learner/prioritizedsweepinglearner.jl, learner/sarsaLnodulatedlearner.jl 
for the core implementation.

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
function getLL_sarsaL_modulated_TParticleFilterJump(data::Array{Int,2},
                                                gamma::Float64, temp::Float64,
                                                surpriseprobability::Float64,
                                                stochasticity::Float64,
                                                nparticles::Union{Int, Float64},
                                                alpha::Float64,
                                                lambda::Float64;
                                                particlefilterseeds = [1,2,3],
                                                use_replacing_trace = true,
                                                init_Q=0.,
                                                initQSA=nothing,
                                                do_debug_print = false
                                                )
    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_sarsaL_modulated_TParticleFilterJump supports only one subject")
    end
    all_puzzles = get_all_puzzles(data)
    LL_seeds = zeros(Float64, length(particlefilterseeds))
    action_selection_probs_seeds = Array{Array{Float64, 1}, 1}(undef, length(particlefilterseeds))
    [action_selection_probs_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    # signals:
    RPE_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [RPE_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    Surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [Surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    γ_surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [γ_surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    QSA_sarsa_modulated_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    Tsas_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    for i_seed in 1:length(particlefilterseeds)
        isendofexperiment = false
        learner = []
        Testimate = []
        action_counter = 0
        for i_puzzle in all_puzzles
            # Puzzle = independent block of the experiment.
            # The code was made for general purposes, in this experiment we
            # have only ONE puzzle.
            Testimate = TParticleFilterJump(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    nparticles = nparticles,
                                    surpriseprobability = surpriseprobability,
                                    stochasticity = stochasticity,
                                    seed = particlefilterseeds[i_seed])
            if initQSA == nothing
                learner = SarsaLModulated(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α = alpha, γ = gamma,
                                    λ = lambda, initvalue = init_Q,
                                    use_replacing_trace = use_replacing_trace
                                    )
            else
                learner = SarsaLModulated(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    α = alpha,
                                    γ = gamma,
                                    λ = lambda,
                                    QSA = copy(initQSA[current_puzzle_id]),
                                    use_replacing_trace = use_replacing_trace
                                    )
            end

            all_epi_in_puzzle = get_all_episodes_in_puzzle(data, i_puzzle)
            for i_episode in all_epi_in_puzzle
                episode_data = get_data_by_puzzle_episode(data, i_puzzle, i_episode)
                path_length = size(episode_data)[1]  # = T
                if do_debug_print
                    println("i_puzzle:$i_puzzle | i_episode:$i_episode | path_length:$path_length")
                end
                for t in 1:path_length # t in THIS episode
                    action_counter += 1
                    state = episode_data[t, Int(c_state)]
                    action = episode_data[t, Int(c_action)]
                    reward = episode_data[t, Int(c_reward)]
                    next_state = episode_data[t, Int(c_nextState)]
                    done = reward>0 ? true : false
                    if do_debug_print
                        println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | done: $done")
                    end

                    p_actions = getSoftmaxProbs(learner.QSA[state,:], temp, [0., 0.])
                    action_selection_probs_seeds[i_seed][action_counter] = copy(p_actions[action])

                    LL_seeds[i_seed] += log(p_actions[action])

                    if do_debug_print
                        println("p(s,a)=$p_actions, action=$action")
                    end

                    # Particle Filtering model learner
                    updatet!(Testimate, state, action, next_state, done)

                    # SarsaL Modulated
                    # Sort out if end of episode or end of experiment
                    next_action = 0
                    if reward==0 #non goal state
                        if nr_samples > action_counter
                            next_action = data[action_counter+1,Int(c_action)]
                            is_surprise_trial = episode_data[t+1, Int(c_surprise_trial)]
                        else
                            # println("i:$i \nS:$state | a:$action | R:$reward | S':$next_state")
                            isendofexperiment = true
                        end
                    end

                    update!(learner, state, action, next_state, next_action, reward,
                            isendofexperiment, (1. - Testimate.γ_surprise[1]))

                    RPE_seeds[i_seed][action_counter] = copy(learner.rpe[1])
                    Surprise_seeds[i_seed][action_counter] = copy(Testimate.Sgm[1])
                    γ_surprise_seeds[i_seed][action_counter] = copy(Testimate.γ_surprise[1])
                    QSA_sarsa_modulated_seeds[i_seed] = copy(learner.QSA)
                    Tsas_seeds[i_seed]= deepcopy(Testimate.Ps1a0s0)

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
        println(finalQSA_sarsa_modulated_record)
        println("action_selection_probs: $action_selection_probs")
        println("LL_mean: $LL_mean")
        println("LL_max: $LL_max")
        println("trial_by_trial_p_sa: $trial_by_trial_p_sa")
    end

    algo = "sarsaL_modulated"
    param_v = [nparticles, gamma, temp, surpriseprobability, stochasticity, alpha, lambda]
    param_str = "nparticles=$nparticles,
                gamma=$(round(gamma,digits=4)), temp=$(round(temp,digits=4)),
                surpriseprobability=$(round(surpriseprobability,digits=4)),
                stochasticity=$(round(stochasticity,digits=4)),
                alpha=$(round(alpha,digits=4)),
                lambda=$(round(lambda,digits=4))"

    results_dict = Dict{String,Any}("finalQSA_sarsa_modulated_record" => QSA_sarsa_modulated_seeds[seed_idx],
                                    "final_Tsas" => Tsas_seeds[seed_idx],
                                    "Surprise" => Surprise_seeds[seed_idx],
                                    "gamma_surprise" => γ_surprise_seeds[seed_idx],
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
