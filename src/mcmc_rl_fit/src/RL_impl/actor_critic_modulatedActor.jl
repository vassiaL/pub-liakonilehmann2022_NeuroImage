"""
Surprise Actor-critic
---------------------

Actor-Critic with a surprise-modulated learning rate for the actor.

See actor-critic.jl for some more implementational info.
See learner/actorcriticlearner.jl and learner/actorcriticmodulatedlearner.jl
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
function getLL_ActorCritic_TDLambda_modulatedActor(data::Array{Int,2},
                                            alpha::Float64,
                                            surpriseprobability::Float64,
                                            stochasticity::Float64,
                                            eta::Float64,
                                            gamma::Float64,
                                            lambda_actor::Float64,
                                            lambda_critic::Float64,
                                            temp::Float64,
                                            nparticles::Union{Int, Float64};
                                            particlefilterseeds = [1,2,3],
                                            init_policy=nothing,
                                            init_Vs = nothing,
                                            use_replacing_trace=true,
                                            do_debug_print = false)
    nparticles = converttoInt(nparticles)
    data_stats, nr_states, nr_actions, nr_samples = get_data_stats_and_dimensions(data)
    if data_stats.nr_subjects != 1
        error("Invalid argument: getLL_ActorCritic_TDLambda_modulatedActor supports only one subject")
    end

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

    finalV_record_seed = Array{Dict{Int,Any}, 1}(undef, length(particlefilterseeds))
    [finalV_record_seed[i_seed] = Dict{Int,Any}() for i_seed in 1:length(particlefilterseeds)]
    final_policyparameters_record_seed = Array{Dict{Int,Any}, 1}(undef, length(particlefilterseeds))
    [final_policyparameters_record_seed[i_seed] = Dict{Int,Any}() for i_seed in 1:length(particlefilterseeds)]

    PolicyParametersImbalance_record_seed  = Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyParametersImbalance_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]
    PolicyImbalance_record_seed = Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyImbalance_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    PolicyParametersDiffChosenVsUnchosen_record_seed =Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyParametersDiffChosenVsUnchosen_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    PolicyDiffChosenVsUnchosen_record_seed = Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyDiffChosenVsUnchosen_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    PolicyParameters_record_seed = Array{Array{Float64,1}, 1}(undef, length(particlefilterseeds))
    [PolicyParameters_record_seed[i_seed] = zeros(Float64, nr_samples) for i_seed in 1:length(particlefilterseeds)]

    for i_seed in 1:length(particlefilterseeds)
        current_puzzle_id = -1
        Testimate = []
        learner = []
        j = 1
        for i = 1:nr_samples
            if current_puzzle_id != data[i,Int(c_puzzle)]
                # Puzzle = independent block of the experiment.
                # The code was made for general purposes, in this experiment we
                # have only ONE puzzle.
                current_puzzle_id = data[i,Int(c_puzzle)]
                if do_debug_print
                    println("new puzzle: $current_puzzle_id")
                end

                # Particle Filter
                Testimate = TParticleFilterJump(nr_states = nr_states,
                                        nr_actions = nr_actions,
                                        nparticles = nparticles,
                                        surpriseprobability = surpriseprobability,
                                        stochasticity = stochasticity,
                                        seed = particlefilterseeds[i_seed])


                if init_policy == nothing # providing init_policy w/o init_Vs is considered a bug
                    learner = ActorCriticModulated(nr_states = nr_states,
                                        nr_actions = nr_actions,
                                        α_critic = eta,
                                        α_actor = alpha,
                                        γ = gamma,
                                        λ_critic = lambda_critic,
                                        λ_actor = lambda_actor,
                                        use_replacing_trace = use_replacing_trace,
                                        do_debug_print = do_debug_print)
                else
                    learner = ActorCriticModulated(nr_states = nr_states,
                                        nr_actions = nr_actions,
                                        α_critic = eta,
                                        α_actor = alpha,
                                        γ = gamma,
                                        λ_critic = lambda_critic,
                                        λ_actor = lambda_actor,
                                        policyparameters = copy(init_policy[current_puzzle_id]),
                                        V = copy(init_Vs[current_puzzle_id]),
                                        use_replacing_trace = use_replacing_trace,
                                        do_debug_print = do_debug_print)
                end
            end
            state = data[i, Int(c_state)]
            action = data[i, Int(c_action)]
            reward::Float64= data[i, Int(c_reward)]
            next_state = data[i, Int(c_nextState)]
            done = reward>0 ? true : false

            p_actions = getSoftmaxProbs(learner.policyparameters[state,:], temp, [0., 0.])

            unbiased_p_sa = p_actions[action]

            action_selection_probs_seeds[i_seed][i] = copy(unbiased_p_sa)
            LL_seeds[i_seed] += log(unbiased_p_sa)

            # Particle Filtering model learner
            updatet!(Testimate, state, action, next_state, done)
            γ_surprise_seeds[i_seed][i] = copy(Testimate.γ_surprise[1])
            Surprise_seeds[i_seed][i] = copy(Testimate.Sgm[1])
            Tsas_seeds[i_seed]= deepcopy(Testimate.Ps1a0s0)
            SPE_seeds[i_seed][i] = copy(Testimate.SPE[1])

            # Track V values (used as fmri regressor)
            V_state_seeds[i_seed][j] = learner.V[state]
            if done
                j+=1
                if j < (nr_samples + data_stats.nr_rewards)
                    V_state_seeds[i_seed][j] = learner.V[next_state] # Track V value of landing state in case it is the Goal
                end
            end
            j+=1

            PolicyParametersImbalance_record_seed[i_seed][i] = abs(learner.policyparameters[state, 1] - learner.policyparameters[state, 2])
            PolicyImbalance_record_seed[i_seed][i] = abs(p_actions[1] - p_actions[2])
            PolicyParametersDiffChosenVsUnchosen_record_seed[i_seed][i] = learner.policyparameters[state, action] - learner.policyparameters[state, mod(action,2)+1]
            PolicyDiffChosenVsUnchosen_record_seed[i_seed][i] = p_actions[action] - p_actions[mod(action,2)+1]
            PolicyParameters_record_seed[i_seed][i] = learner.policyparameters[state, action]

            # Actor-Critic
            update!(learner, state, action, next_state, reward, unbiased_p_sa, (1. - Testimate.γ_surprise[1]))
            TD_seeds[i_seed][i] = copy(learner.td_error[1])

            finalV_record_seed[i_seed][current_puzzle_id] = copy(learner.V)
            final_policyparameters_record_seed[i_seed][current_puzzle_id] = copy(learner.policyparameters)

        end
    end

    LL = get_meanLL_acrossseeds(LL_seeds)

    action_selection_probs_cat = hcat(action_selection_probs_seeds...)
    action_selection_probs = mean(action_selection_probs_cat, dims=2)[:]
    trial_by_trial_p_sa = (action_selection_probs)

    # For signals: return the ones corresponding to the Max LL across seeds
    LL_max, seed_idx = findmax(LL_seeds)

    TD_record = copy(TD_seeds[seed_idx])
    SPE_record = copy(SPE_seeds[seed_idx])
    trial_by_trial_p_sa_max = (action_selection_probs_seeds[seed_idx])

    trace_type = use_replacing_trace ? "ReplTrc" : "EligTrc"
    algo = "Actor-Critic (TD-Lambda, $trace_type) modulated Actor"
    param_v = [alpha, surpriseprobability, stochasticity, eta, gamma, lambda_actor, lambda_critic, temp, nparticles]
    param_str = "alpha(actor)=$(round(alpha,digits=4)),
                surpriseprobability=$(round(surpriseprobability,digits=4)),
                stochasticity=$(round(stochasticity,digits=4)),
                eta(critic)=$(round(eta,digits=4)),
                gamma=$(round(gamma,digits=4)),
                lambda_actor=$(round(lambda_actor,digits=4)),
                lambda_critic=$(round(lambda_critic,digits=4)),
                temp=$(round(temp,digits=4)),
                nparticles=$nparticles,
                init_policy=$(repr(init_policy)),
                init_Vs=$(repr(init_Vs)),
                use_replacing_trace=$use_replacing_trace"

    results_dict = Dict{String,Any}(
        "trialByTrial_Vs" => V_state_seeds[seed_idx],
        "final_V" => finalV_record_seed[seed_idx],
        "final_PolicyParameters" => final_policyparameters_record_seed[seed_idx],
        "final_Tsas" => Tsas_seeds[seed_idx],
        "Surprise" => Surprise_seeds[seed_idx],
        "gamma_surprise" => γ_surprise_seeds[seed_idx],
        "nparticles" => nparticles,
        "particlefilterseeds" => particlefilterseeds,
        "trial_by_trial_p_sa_max" => trial_by_trial_p_sa_max,
        "PolicyParametersImbalance" => PolicyParametersImbalance_record_seed[seed_idx],
        "PolicyImbalance" => PolicyImbalance_record_seed[seed_idx],
        "PolicyParametersDiffChosenVsUnchosen" => PolicyParametersDiffChosenVsUnchosen_record_seed[seed_idx],
        "PolicyDiffChosenVsUnchosen" => PolicyDiffChosenVsUnchosen_record_seed[seed_idx],
        "PolicyParameters" => PolicyParameters_record_seed[seed_idx],
        )

    signals = Step_by_step_signals(
    data_stats.all_subject_ids[1],
    TD_record,  # RPE
    SPE_record,  # no SPE
    trial_by_trial_p_sa,
    LL,
    data_stats,
    algo,
    param_v,
    param_str,
    99, # dummy value
    results_dict)
    return LL, signals
end
