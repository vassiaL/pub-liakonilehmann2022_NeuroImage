using Distributed
using Distributions
using PyPlot


function playWithData(;subj_id = 3
                        #, i_episode = 20
                        )

    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEIC_all_fMRI.csv"
    max_nr_episodes = 50

    all_data, all_data_stats = RL_Fit.load_SARSPEI(data_path; max_episodes_per_subj = max_nr_episodes)

    subj_data = RL_Fit.filter_by_person(all_data, subj_id)

    all_puzzles = RL_Fit.get_all_puzzles(subj_data)
    i_puzzle = 1

    all_epi_in_puzzle = RL_Fit.get_all_episodes_in_puzzle(subj_data, i_puzzle)

    episode_data = []
    for i_episode in all_epi_in_puzzle
        episode_data = RL_Fit.get_data_by_puzzle_episode(subj_data, i_puzzle, i_episode)
        for i in 1:size(episode_data, 1)
            @show episode_data[i, :]
        end
        println("------")
    end
    # episode_data
end

# NOTE:
# ----
# Row coding of data file:
#SARSPE_COLS
#c_state=1
#c_action=2
#c_reward=3
#c_nextState=4
#c_puzzle=5
#c_episode=6
#c_subject=7
#c_surprise_trial=8
#c_RT_prctile=9
#c_action_value=10
#c_graph_version=11

function get_nr_episodes()
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"
    max_episodes_per_subj = 70

    all_data, all_stats = RL_Fit.load_SARSPEI(data_path;max_episodes_per_subj=max_episodes_per_subj)
    all_subject_ids = all_stats.all_subject_ids

    for subj_id in all_subject_ids
        println("--------")
        @show subj_id

        subj_data = RL_Fit.filter_by_person(all_data, subj_id)
        graph_id = subj_data[1,Int(RL_Fit.c_graph_version)]
        @show graph_id
        nr_states = size(subj_data, 1)
        @show nr_states
        nr_surprise_trials = sum(subj_data[:, Int(RL_Fit.c_surprise_trial)])
        @show nr_surprise_trials
        @show nr_surprise_trials/nr_states
        all_episodes = unique(subj_data[:,Int(RL_Fit.c_episode)])
        @show maximum(all_episodes)
    end
end

function get_someinfonow()
    # Get some info about all subjects

    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"
    max_episodes_per_subj = 80
    (all_data, all_stats) = RL_Fit.load_SARSPEI(data_path;max_episodes_per_subj=max_episodes_per_subj)
    all_subject_ids = all_stats.all_subject_ids
    # println("all_subject_ids: $all_subject_ids")

    graph_ids = zeros(Int64, length(all_subject_ids))
    nr_episodes = zeros(Int64, length(all_subject_ids))
    nr_actions = zeros(Int64, length(all_subject_ids))

    for subj_id in all_subject_ids
        @show subj_id
        subj_data = RL_Fit.filter_by_person(all_data, subj_id)

        graph_id = subj_data[1,Int(RL_Fit.c_graph_version)]
        @show graph_id
        graph_ids[subj_id] = copy(graph_id)

        all_episodes = unique(subj_data[:,Int(RL_Fit.c_episode)])
        @show maximum(all_episodes)
        nr_episodes[subj_id] = copy(maximum(all_episodes))

        actions = subj_data[:,Int(RL_Fit.c_action)]
        @show length(actions)
        nr_actions[subj_id] = copy(length(actions))

        println("-------------------")
    end
    graph_ids, nr_episodes, nr_actions
end

function singleSubjectData(;id_subject=3)
    # Using an algorithm (edit: learner_function_V below) and a set of parameters
    # (edit params below) go through the actions of one subject (id_subject)
    # and obtain the log-likelihood and the corresponding learning signals
    # (eg RPE, SPE...)
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEIC_all_fMRI.csv"

    data_single_subj, data_stats = RL_Fit.load_SARSPEI(data_path; subject_id = id_subject)
    # -------
    # Surprise Actor-critic
    # ---------------------
    learner_function_V = RL_Fit.getLL_ActorCritic_TDLambda_modulatedActor_V
    params = [0.79724,  #alpha
    0.00433, #surpriseprobability
    2.78163, #stochasticity
    3.24411e-5, #eta
    0.99664, #gamma
    0.79392,  #lambda_actor
    0.30497,  #lambda_critic
    0.39196, # temp
    15, #nparticles
    ]
    # -------
    # SARSA-λ
    # -------
    # learner_function_V = RL_Fit.getLL_SarsaLambda_V
    # params = [0.2, #alpha
    # 0.9999, #gamma
    # 0.61, #lambda
    # 0.25,# temp
    # ]
    # -------
    # REINFORCE
    # ---------
    # learner_function_V = RL_Fit.getLL_REINFORCE_V
    # params = [0.96, #alpha
    # 0.76, #gamma
    # 0.95,# temp
    # ]
    # # -------
    # Actor-critic
    # ------------
    # learner_function_V = RL_Fit.getLL_ActorCritic_TDLambda_V
    # params = [0.5009131276460307, #alpha
    # 4.6834146251373146e-5, #eta
    # 0.8205988691213357, #gamma
    # 0.9599330699486914,#lambda actor
    # 0.5782935181795414,#lambda critic
    # 0.24363820949361994# temp
    # ]
    # -------
    # -------
    println("hi")
    @show learner_function_V
    LL, signals = learner_function_V(data_single_subj, params)
    signals=signals[id_subject]

    LL, signals
end



function lookatpolicyimbalances(signalsArray;
                                labels = ["REINFORCE", "Actor-critic", "SARSA"],
                                temperature_index = [3, 6, 4],
                                # save_path = "./"
                                subjID = 3
                                )
    # Plot policy imbalances (similar to Fig. A.3)
    # signalsArray = array with signals of different algorithms
    #                run on the data of one subject (subjID).
    #                (ie output of some learner_function_V,
    #                for example RL_Fit.getLL_ActorCritic_TDLambda_V)
    # temperature_index = index of temperature parameter in the list of parameters
    #                   of the algorithm (look at order in eg RL_Fit.getLL_ActorCritic_TDLambda_V)

    save_path = makehomesavepath("test_policyImbalance_")
    mkdir(save_path)

    fig, ax = subplots(nrows=3, ncols=1, figsize=(13, 19))
    ax_idx = 0

    # ---- f(st,at_chosen) - f(st,at_notchosen)
    ax_idx += 1
    for i in 1:length(signalsArray)
        ax[ax_idx].scatter(1:length(signalsArray[i].results_dict["PolicyParametersDiffChosenVsUnchosen"]),
                    signalsArray[i].results_dict["PolicyParametersDiffChosenVsUnchosen"],
                    label=labels[i])
    end
    ax[ax_idx].set_title(L"$f(s_t, a_{chosen}) - f(s_t, a_{not chosen})$")
    ax[ax_idx].set_xlabel("actions")
    ax[ax_idx].grid()
    ax[ax_idx].legend()

    # ---- f(st,at_chosen) - f(st,at_notchosen) / τ
    ax_idx += 1
    for i in 1:length(signalsArray)
        temperature = signalsArray[i].param_vector[temperature_index[i]]
        # @show temperature
        ax[ax_idx].scatter(1:length(signalsArray[i].results_dict["PolicyParametersDiffChosenVsUnchosen"]),
                    signalsArray[i].results_dict["PolicyParametersDiffChosenVsUnchosen"] ./ temperature,
                    label=labels[i])
    end
    ax[ax_idx].set_title(L"$(f(s_t, a_{chosen}) - f(s_t, a_{not chosen})) / \tau $")
    ax[ax_idx].set_xlabel("actions")
    ax[ax_idx].grid()
    ax[ax_idx].legend()

    # ---- π(st,at_chosen) - π(st,at_notchosen)
    # fig = figure(); ax = gca()
    ax_idx += 1
    for i in 1:length(signalsArray)
        ax[ax_idx].scatter(1:length(signalsArray[i].results_dict["PolicyDiffChosenVsUnchosen"]),
                    signalsArray[i].results_dict["PolicyDiffChosenVsUnchosen"],
                    label=labels[i])
    end
    ax[ax_idx].set_title(L"$\pi(s_t, a_{chosen}) - \pi(s_t, a_{not chosen})$")
    ax[ax_idx].set_xlabel("actions")
    ax[ax_idx].grid()
    ax[ax_idx].legend()

    fig.tight_layout()
    fig.savefig(joinpath(save_path, "subj_"*string(subjID)*"PolImbalances_.pdf"))
    plt.close()
end

function plotscattersignals(signalsArray;
                                labels = ["REINFORCE",
                                        "Actor-critic",
                                        "SARSA"],
                                save_path = "./",
                                subjID = 3
                                )
    # Plot some signals.
    # signalsArray = array with signals of different algorithms
    # run on the data of one subject (subjID).
    # (ie output of some learner_function_V)
    # NOTE: this function is just for illustration and playing around!
    # Need to edit which signals you would like to plot.

    # ---- SPE
    fig = figure(); ax = gca()
    for i in 1:length(signalsArray)
        ax.scatter(1:length(signalsArray[i].SPE),
                    signalsArray[i].SPE,
                    label=labels[i])
    end
    ax.set_title("SPE")
    ax.set_xlabel("actions")
    ax.grid()
    ax.legend()

    # --- Surprise
    fig2 = figure(); ax2 = gca()
    for i in 1:length(signalsArray)
        if  signalsArray[i].algorithm_name == "hybrid_actorcritic_FWD"
            ax2.scatter(1:length(signalsArray[i].SPE),
                        signalsArray[i].SPE,
                        label=labels[i])
        else
            ax2.scatter(1:length(signalsArray[i].results_dict["Surprise"]),
                        signalsArray[i].results_dict["Surprise"],
                        label=labels[i])
        end
    end
    ax2.set_title(L"$S_{BF}$")
    ax2.set_xlabel("actions")
    ax2.grid()
    ax2.legend()

end
