"""
Surprise-REINFORCE
------------------

REINFORCE with surprise-modulated learning rate

See REINFORCE.jl for some more implementational info
See learner/REINFORCEmodulatedlearner.jl for the core implementation.

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
function getLL_REINFORCE_modulated(data::Array{Int,2},
                                    alpha::Float64,
                                    surpriseprobability::Float64,
                                    stochasticity::Float64,
                                    gamma::Float64,
                                    temp::Float64,
                                    nparticles::Union{Int, Float64};
                                    particlefilterseeds = [1,2,3],
                                    do_debug_print = false)
    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_REINFORCE_modulated supports only one subject")
    end
    if nr_actions!=2; error("REINFORCE supports two actions only"); end
    all_puzzles = get_all_puzzles(data)

    action_selection_probs_seeds = Array{Array{Float64, 1}, 1}(undef, length(particlefilterseeds))
    [action_selection_probs_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    LL_seeds = zeros(Float64, length(particlefilterseeds))

    theta_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    Tsas_seeds = Array{Any, 1}(undef, length(particlefilterseeds))
    #signals:
    Surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [Surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    γ_surprise_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [γ_surprise_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    SPE_seeds =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [SPE_seeds[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    PolicyParametersDiffChosenVsUnchosen_record_seed =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyParametersDiffChosenVsUnchosen_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    PolicyDiffChosenVsUnchosen_record_seed = Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyDiffChosenVsUnchosen_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    finaltheta_record = 0
    for i_seed in 1:length(particlefilterseeds)
        action_counter = 0
        Testimate = []
        learner = []
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

            learner = REINFORCEModulated(nr_states = nr_states,
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

                    LL_seeds[i_seed] += log(p_actions[action])
                    action_selection_probs_seeds[i_seed][action_counter] = copy(p_actions[action])

                    # Particle Filtering model learner
                    updatet!(Testimate, state, action, next_state, done)
                    γ_surprise_seeds[i_seed][action_counter] = copy(Testimate.γ_surprise[1])
                    Surprise_seeds[i_seed][action_counter] = copy(Testimate.Sgm[1])
                    Tsas_seeds[i_seed]= deepcopy(Testimate.Ps1a0s0)
                    SPE_seeds[i_seed][action_counter] = copy(Testimate.SPE[1])

                    PolicyParametersDiffChosenVsUnchosen_record_seed[i_seed][action_counter] = learner.θ[state, action] - learner.θ[state, mod(action,2)+1]
                    PolicyDiffChosenVsUnchosen_record_seed[i_seed][action_counter] = p_actions[action] - p_actions[mod(action,2)+1]

                    # REINFORCE modulated
                    update!(learner, state, action, reward, p_actions[action], (1. - Testimate.γ_surprise[1]))

                    if do_debug_print
                        println("t:$t | s:$state | a:$action | R:$reward | s':$next_state | p(s,a):$(round(p_action,digits=2))")
                    end
                end # path_length (in single episode)
          end # episode
      end # puzzle
      theta_seeds[i_seed] = copy(learner.θ)
  end # seed

  LL = get_meanLL_acrossseeds(LL_seeds)

  action_selection_probs_cat = hcat(action_selection_probs_seeds...)
  action_selection_probs = mean(action_selection_probs_cat, dims=2)[:]
  trial_by_trial_p_sa = (action_selection_probs)

  # For signals: return the ones corresponding to the Max LL across seeds
  LL_max, seed_idx = findmax(LL_seeds)
  trial_by_trial_p_sa_max = (action_selection_probs_seeds[seed_idx])

  SPE_record = copy(SPE_seeds[seed_idx])

  algo = "REINFORCE_modulated"
  param_v = [alpha, surpriseprobability, stochasticity, gamma, temp, nparticles]
  param_str = "alpha=$(round(alpha,digits=4)),
              surpriseprobability=$(round(surpriseprobability,digits=4)),
              stochasticity=$(round(stochasticity,digits=4)),
              gamma=$(round(gamma,digits=4)),
              temp=$(round(temp,digits=4)),
              nparticles=$nparticles"

  results_dict = Dict{String,Any}("final_Theta" => theta_seeds[seed_idx],
                                "final_Tsas" => Tsas_seeds[seed_idx],
                                "Surprise" => Surprise_seeds[seed_idx],
                                "gamma_surprise" => γ_surprise_seeds[seed_idx],
                                "nparticles" => nparticles,
                                "particlefilterseeds" => particlefilterseeds,
                                "trial_by_trial_p_sa_max" => trial_by_trial_p_sa_max,
                                "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record_seed[seed_idx],
                                "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record_seed[seed_idx],)

  signals = Step_by_step_signals(
    data_stats.all_subject_ids[1],
    [], # no RPE
    SPE_record, # no SPE
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
