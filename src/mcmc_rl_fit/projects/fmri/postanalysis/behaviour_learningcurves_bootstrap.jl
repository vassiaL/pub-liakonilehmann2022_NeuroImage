using Distributions
using PyPlot
using LsqFit
using CSV, DataFrames, DataFramesMeta
using StatsBase

""" Functions for results of Fig. 2E """
# Run:
# mean_thrs_delay_1, mean_thrs_delay_3 = SinergiafMRI_datafit.get_state_visits_bootstrapped()
function get_state_visits_bootstrapped(;
                                        # filestr = "_Sim_",
                                        filestr = ""
                                        )
    # Real:
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"
    # Simulated:
    # data_path = "./mcmc_rl_fit/projects/fmri/data_sim/SARSPEICZVG_all_fMRI_Sim_exporder_SurpAC_run2.csv"

    max_episodes_per_subj = 40
    (all_data, all_stats) = RL_Fit.load_SARSPEI(data_path;max_episodes_per_subj=max_episodes_per_subj)
    all_subject_ids = all_stats.all_subject_ids
    println("all_subject_ids: $all_subject_ids")

    nrOfBootstrapRounds = 200

    threshold_delay_matrices_1 = Array{Array{Float64,1}, 1}(undef, nrOfBootstrapRounds)
    [threshold_delay_matrices_1[i] = zeros(Float64, 7) for i in 1:nrOfBootstrapRounds]
    threshold_delay_matrices_3 = Array{Array{Float64,1}, 1}(undef, nrOfBootstrapRounds)
    [threshold_delay_matrices_3[i] = zeros(Float64, 7) for i in 1:nrOfBootstrapRounds]
    #
    # threshold_delay_matrices_nofitting_1 = Array{Array{Float64,1}, 1}(undef, nrOfBootstrapRounds)
    # [threshold_delay_matrices_nofitting_1[i] = zeros(Float64, 7) for i in 1:nrOfBootstrapRounds]
    # threshold_delay_matrices_nofitting_3 = Array{Array{Float64,1}, 1}(undef, nrOfBootstrapRounds)
    # [threshold_delay_matrices_nofitting_3[i] = zeros(Float64, 7) for i in 1:nrOfBootstrapRounds]

    graph_1_subjIDs = [4, 5, 6, 8, 10, 12, 15, 16, 19, 21]
    graph_3_subjIDs = [1, 2, 3, 7, 9, 11, 13, 14, 17, 18, 20]
    for i in 1:nrOfBootstrapRounds
        # Sample with replacement

        bootstrapped_subject_ids_graph_1 = sample(graph_1_subjIDs, length(graph_1_subjIDs), replace=true)
        bootstrapped_subject_ids_graph_3 = sample(graph_3_subjIDs, length(graph_3_subjIDs), replace=true)
        bootstrapped_subject_ids = vcat(bootstrapped_subject_ids_graph_1, bootstrapped_subject_ids_graph_3)

        # Get graph_visits
        graph_1_visits, graph_3_visits = get_state_visits_singleround(all_data,
                        bootstrapped_subject_ids, max_episodes_per_subj = max_episodes_per_subj)

        threshold_delay_matrices_1[i] = get_state_fittedthreshold(1, graph_1_visits)
        threshold_delay_matrices_3[i] = get_state_fittedthreshold(3, graph_3_visits)

        # threshold_delay_matrices_nofitting_1[i] = get_state_threshold_nofitting(1, graph_1_visits)
        # threshold_delay_matrices_nofitting_3[i] = get_state_threshold_nofitting(3, graph_3_visits)

    end

    mean_thrs_delay_1, std_thrs_delay_1, stderror_thrs_delay_1 =  get_bootstrapped_stats(threshold_delay_matrices_1)
    mean_thrs_delay_3, std_thrs_delay_3, stderror_thrs_delay_3 =  get_bootstrapped_stats(threshold_delay_matrices_3)

    write_bootstrapped_results_csv(1, mean_thrs_delay_1, std_thrs_delay_1, stderror_thrs_delay_1, filestr = filestr)
    write_bootstrapped_results_csv(3, mean_thrs_delay_3, std_thrs_delay_3, stderror_thrs_delay_3, filestr = filestr)

    # ------------------------------------------
    # mean_thrs_delay_nofitting_1, std_thrs_delay_nofitting_1, stderror_thrs_delay_nofitting_1 =  get_bootstrapped_stats(threshold_delay_matrices_nofitting_1)
    # mean_thrs_delay_nofitting_3, std_thrs_delay_nofitting_3, stderror_thrs_delay_nofitting_3 =  get_bootstrapped_stats(threshold_delay_matrices_nofitting_3)
    #
    # write_bootstrapped_results_csv(1, mean_thrs_delay_nofitting_1, std_thrs_delay_nofitting_1, stderror_thrs_delay_nofitting_1, filestr=filestr*"nofitting_")
    # write_bootstrapped_results_csv(3, mean_thrs_delay_nofitting_3, std_thrs_delay_nofitting_3, stderror_thrs_delay_nofitting_3, filestr=filestr*"nofitting_")

    mean_thrs_delay_1, mean_thrs_delay_3#,  mean_thrs_delay_nofitting_1, mean_thrs_delay_nofitting_3
end

""" Similar to get_state_visits() """
function get_state_visits_singleround(all_data, subject_ids; max_episodes_per_subj = 40)

    graph_1_visits = zeros(Int64, max_episodes_per_subj,7, 2)
    graph_3_visits = zeros(Int64, max_episodes_per_subj,7, 2)
    # @show subject_ids
    for subj_id in subject_ids
        # @show subj_id
        subj_data = RL_Fit.filter_by_person(all_data, subj_id)
        graph_id = subj_data[1,Int(RL_Fit.c_graph_version)]
        all_episodes = unique(subj_data[:,Int(RL_Fit.c_episode)])
        trajectory_count_per_state = zeros(Int64, 7) #count for each state individually how many times it participated in a rewarded episode
        # per subject, do not count (sum), just set to 0 or 1.
        state_action_indicator = zeros(Int64, max_episodes_per_subj, 7, 2)

        for episode in all_episodes
            for s in 2:7
                # check if state s has "contributed" to the current episode. if yes, count this trajectory.
                if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) )
                    trajectory_count_per_state[s] += 1
                    # now count the value of actions. distinguish two cases:
                    # for some states, the two actions are equally valid. In this case count the action
                    # for other states, there is a correct (1) and a wrong(-1) action
                    # action correct:
                    if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== 0))
                        # action A or action B?
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action)] .== 1))
                            # A
                            state_action_indicator[trajectory_count_per_state[s], s, 1] = 1 # do not +=1
                        end
                        # note: it's not an ELSE here. we check for both and count both options at most once
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action)] .== 2))
                            # B
                            state_action_indicator[trajectory_count_per_state[s], s, 2] = 1 # do not +=1
                        end
                    else
                        # action correct or wrong?
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== +1))
                            # correct
                            state_action_indicator[trajectory_count_per_state[s], s, 2] = 1 # do not +=1
                        end
                        # note: it's not an ELSE here. we check for both and count both options at most once
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== -1))
                            # wrong
                            state_action_indicator[trajectory_count_per_state[s], s, 1] = 1 # do not +=1
                        end
                    end
                end
            end # end all states
        end # end all episodes
        # @show state_action_indicator
        if graph_id == 1
            graph_1_visits += state_action_indicator
        elseif graph_id == 3
            graph_3_visits += state_action_indicator
        else
            error("unknown graph id: $graph_id")
        end
    end # end all subject
    graph_1_visits, graph_3_visits
end

function get_state_fittedthreshold(graph_id::Int,
                            graph_state_counts;
                            learning_threshold=0.8,
                            min_visits_requirement = 2,
                            savepath = "/Users/vasia/Desktop/")
    dfresults = []
    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    init_vals_exp = [0.5, 0.]
    threshold_delay_matrix = zeros(length(states)+1)
    for s = states
        # println(s)
        counts_per_episode_actionwrong = vec(graph_state_counts[:,s,1])
        counts_per_episode_actioncorrect = vec(graph_state_counts[:,s,2])
        summed_counts_per_episode = counts_per_episode_actionwrong.+counts_per_episode_actioncorrect
        episodes_min_visits = findall(summed_counts_per_episode .>= min_visits_requirement)
        # @show episodes_min_visits
        ratio = counts_per_episode_actioncorrect[episodes_min_visits] ./ summed_counts_per_episode[episodes_min_visits]
        weight = (summed_counts_per_episode[episodes_min_visits])
        nr_x = length(ratio)

        x_data = 1:nr_x
        y_data = ratio
        # markersize = 1+2*sqrt(weight[i]),
        threshold_delay = 0.
        weight = Float64.(weight)
        if minimal_dist[1,s] != minimal_dist[2,s]
            fit = curve_fit(model_exp, x_data, y_data, weight, init_vals_exp) # Weighted cost function
            y_hat = model_exp(x_data, fit.param)
            threshold_delay = inv_model_exp(learning_threshold, fit.param)
        end
        if threshold_delay < 0.
            if fit.param[2] > 0.
                threshold_delay = 0.
            else
                threshold_delay = Float64(episodes_min_visits[end])
            end
            # @show threshold_delay
            # println("----------")
        end
        if threshold_delay > Float64(episodes_min_visits[end])
            threshold_delay = Float64(episodes_min_visits[end])
        end
        threshold_delay_matrix[s] = threshold_delay
    end
    threshold_delay_matrix
end

# NOTE: NOT USED!!!
function get_state_threshold_nofitting(graph_id::Int,
                            graph_state_counts;
                            learning_threshold=0.8,
                            min_visits_requirement = 2,
                            savepath = "/Users/vasia/Desktop/")
    dfresults = []
    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    init_vals_exp = [0.5, 0.]
    threshold_delay_matrix = zeros(length(states)+1)
    for s = states
        # println(s)
        counts_per_episode_actionwrong = vec(graph_state_counts[:,s,1])
        counts_per_episode_actioncorrect = vec(graph_state_counts[:,s,2])
        summed_counts_per_episode = counts_per_episode_actionwrong.+counts_per_episode_actioncorrect
        episodes_min_visits = findall(summed_counts_per_episode .>= min_visits_requirement)
        # @show episodes_min_visits
        ratio = counts_per_episode_actioncorrect[episodes_min_visits] ./ summed_counts_per_episode[episodes_min_visits]
        weight = (summed_counts_per_episode[episodes_min_visits])
        nr_x = length(ratio)

        # markersize = 1+2*sqrt(weight[i]),
        threshold_delay = 0.
        weight = Float64.(weight)
        if minimal_dist[1,s] != minimal_dist[2,s]
            threshold_delay = findfirst(x-> x.>=0.8, ratio)
        end
        if isnothing(threshold_delay)
            threshold_delay = Float64(episodes_min_visits[end])
        end
        threshold_delay_matrix[s] = threshold_delay
    end
    threshold_delay_matrix
end


function get_bootstrapped_stats(threshold_delay_matrices)
    thrs_delay_cat = hcat(threshold_delay_matrices...)
    mean_thrs_delay = mean(thrs_delay_cat, dims=2)
    std_thrs_delay = std(thrs_delay_cat, dims=2)
    @show size(threshold_delay_matrices,1)
    stderror_thrs_delay = std(thrs_delay_cat, dims=2) ./ sqrt(size(threshold_delay_matrices,1))
    mean_thrs_delay, std_thrs_delay, stderror_thrs_delay
end


function write_bootstrapped_results_csv(graph_id::Int64,
                                        mean_thrs_delay,
                                        std_thrs_delay,
                                        stderror_thrs_delay;
                                        filestr="",
                                        # filestr = "_Sim_",
                                        # savepath = "./"
                                        )
    savepath = makehomesavepath("learningcurvebars_bootstrap")
    mkdir(savepath)

    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    dfresults = DataFrame()
    for s = states
        min_dist = sort(minimal_dist[:,s] .- 1.)
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay_mean")] = [mean_thrs_delay[s]]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay_std")] = [std_thrs_delay[s]]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay_stderror")] = [stderror_thrs_delay[s]]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_dist1vsdist2")] = [string(Int(min_dist[1]))*"-or-"*string(Int(min_dist[2]))]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_graphid")] = [graph_id]
    end
    CSV.write(joinpath(savepath, "learningcurvebar_bootstrap_"*filestr*"G" * string(graph_id) *".csv"), dfresults,
        delim = " ");
    # dfresults
end
