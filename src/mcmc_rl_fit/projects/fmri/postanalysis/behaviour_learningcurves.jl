using Distributions
using PyPlot
using LsqFit
using CSV, DataFrames, DataFramesMeta

""" Functions for results of Fig. 2D """
# Run:
# graph_1_visits, graph_3_visits, graph_1_subjIDs, graph_3_subjIDs = SinergiafMRI_datafit.get_state_visits()
# and then (to process and save results in csv):
# SinergiafMRI_datafit.csv_state_decision_stats(1, graph_1_visits)
# SinergiafMRI_datafit.csv_state_decision_stats(3, graph_3_visits)
#
# and (for quick plotting) :
# SinergiafMRI_datafit.plot_state_decision_stats(1, graph_1_visits)
# SinergiafMRI_datafit.plot_state_decision_stats(3, graph_3_visits)
function get_state_visits()
    # Real:
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"
    # Simulated:
    # data_path = "./mcmc_rl_fit/projects/fmri/data_sim/SARSPEICZVG_all_fMRI_Sim_exporder_SurpAC_run2.csv"

    max_episodes_per_subj = 40
    (all_data, all_stats) = RL_Fit.load_SARSPEI(data_path;max_episodes_per_subj=max_episodes_per_subj)
    all_subject_ids = all_stats.all_subject_ids
    println("all_subject_ids: $all_subject_ids")

    graph_1_visits = zeros(Int64, max_episodes_per_subj,7, 2)
    graph_3_visits = zeros(Int64, max_episodes_per_subj,7, 2)
    graph_1_subjIDs = []
    graph_3_subjIDs = []
    for subj_id in all_subject_ids
        @show subj_id
        subj_data = RL_Fit.filter_by_person(all_data, subj_id)
        graph_id = subj_data[1,Int(RL_Fit.c_graph_version)]
        all_episodes = unique(subj_data[:,Int(RL_Fit.c_episode)])
        trajectory_count_per_state = zeros(Int64, 7) #count for each state individually how many times it participated in a rewarded episode
        # per subject, do not count (sum), just set to 0 or 1.
        state_action_indicator = zeros(Int64, max_episodes_per_subj, 7, 2)

        for episode in all_episodes
#             print("\nS:$subj_id/E:$episode ")
            for s in 2:all_stats.max_state_id
                # check if state s has "contributed" to the current episode. if yes, count this trajectory.
                if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) )
                    trajectory_count_per_state[s] += 1
                    #             end
                    # now count the value of actions. distinguish two cases:
                    # for some states, the two actions are equally valid. In this case count the action
                    # for other states, there is a correct (1) and a wrong(-1) action
                    # action correct:
                    if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== 0))
                        # action A or action B?
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action)] .== 1))
                            # A
                            state_action_indicator[trajectory_count_per_state[s], s, 1] = 1 # do not +=1
                            #                         print("A")
                        end
                        # note: it's not an ELSE here. we check for both and count both options at most once
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action)] .== 2))
                            # B
                            state_action_indicator[trajectory_count_per_state[s], s, 2] = 1 # do not +=1
                            #                         print("B")
                        end
                    else
                        # action correct or wrong?
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== +1))
                            # correct
                            state_action_indicator[trajectory_count_per_state[s], s, 2] = 1 # do not +=1
                            #                         print("c")
                        end
                        # note: it's not an ELSE here. we check for both and count both options at most once
                        if any( (subj_data[:,Int(RL_Fit.c_episode)] .== episode) .& (subj_data[:,Int(RL_Fit.c_state)] .== s) .& (subj_data[:,Int(RL_Fit.c_action_value)] .== -1))
                            # wrong
                            state_action_indicator[trajectory_count_per_state[s], s, 1] = 1 # do not +=1
                            #                         print("w")
                        end
                    end
                else
#                     print("N")
                end
            end # end all states
        end # end all episodes
        if graph_id == 1
            graph_1_visits += state_action_indicator
            push!(graph_1_subjIDs, subj_id)
        elseif graph_id == 3
            graph_3_visits += state_action_indicator
            push!(graph_3_subjIDs, subj_id)
        else
            error("unknown graph id: $graph_id")
        end
    end # end all subject
    graph_1_visits, graph_3_visits, graph_1_subjIDs, graph_3_subjIDs
end

model_exp(episode, param) = 1. .- param[1] * exp.(-Float64.(episode).*param[2])
# model_exp(episode, param) = 1- param[1]*exp.(-episode.*param[2])
model_lin(episode, param) = param[1] .+ episode .* param[2]
inv_model_exp(val, param) = (-1/param[2])*log((1-val)/param[1])

function get_this_graph_info(graph_info, graph_id)
    if graph_id == 1
        states = graph_info["state_order_g1"]
        minimal_dist = graph_info["minimal_dist_g1"]
        state_txt = graph_info["state_txt_g1"]
        graph = graph_info["transition_matrix_g1"]
    else
        states = graph_info["state_order_g3"]
        minimal_dist = graph_info["minimal_dist_g3"]
        state_txt = graph_info["state_txt_g3"]
        graph = graph_info["transition_matrix_g3"]
    end
    states, minimal_dist, state_txt, graph
end
function plot_state_decision_stats(graph_id::Int,
                            graph_state_counts;
                            learning_threshold=0.8,
                            min_visits_requirement = 2)

    save_path = makehomesavepath("learningcurves")
    mkdir(save_path)

    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    y_min_lim = 0.25
    init_vals_lin = [0.5, 0.01]
    init_vals_exp = [0.5, 0.]
    fig, axes = subplots(nrows=6, ncols=1, figsize=(8.5, 11.5))
    ax_idx = 0
    for s = states
        println(s)
        ax_idx += 1

        counts_per_episode_actionwrong = vec(graph_state_counts[:,s,1])
        counts_per_episode_actioncorrect = vec(graph_state_counts[:,s,2])
        summed_counts_per_episode = counts_per_episode_actionwrong.+counts_per_episode_actioncorrect
        episodes_min_visits = findall(summed_counts_per_episode .>= min_visits_requirement)
        ratio = counts_per_episode_actioncorrect[episodes_min_visits] ./ summed_counts_per_episode[episodes_min_visits]
        weight = (summed_counts_per_episode[episodes_min_visits])
        nr_x = length(ratio)

        x_data = 1:nr_x
        y_data = ratio
        marker_symbol = (minimal_dist[1,s] == minimal_dist[2,s]) ? "oy" : "og"
        for i in 1:nr_x #plot does not accept a vector for size.
            axes[ax_idx].plot(x_data[i], ratio[i], marker_symbol,
                                            markersize = 1+2*sqrt(weight[i]),
                                            markeredgewidth=0.0)
        end
        weight = 1.0 .*weight  # just one example why I sometimes do not like JuliaLang
        # weight = Float64.(weight)
        if minimal_dist[1,s] == minimal_dist[2,s]
            fit = curve_fit(model_lin, x_data, y_data, weight, init_vals_lin)
            y_hat = model_lin(x_data, fit.param)
            axes[ax_idx].plot(x_data, y_hat, "-r", lineWidth=2)
            axes[ax_idx].set_title("G: $graph_id | S: $(state_txt[s]) | dist:($(minimal_dist[1,s])/$(minimal_dist[2,s]))| LIN fit:$(round.(fit.param, digits=2))", fontsize=10)
            axes[ax_idx].set_yticks((0,1))
            axes[ax_idx].set_yticklabels(("A", "B"))
            axes[ax_idx].set_ylabel("Decision", fontsize=10)
            axes[ax_idx].set_ylim([-0.05, 1.05])

        else
            fit = curve_fit(model_exp, x_data, y_data, weight, init_vals_exp) # Weighted cost function
            y_hat = model_exp(x_data, fit.param)
            axes[ax_idx].plot(x_data, y_hat, "-r", lineWidth=2)
            threshold_delay = inv_model_exp(learning_threshold, fit.param)
            @show s
            @show fit.param
            @show threshold_delay
            if -10< threshold_delay && threshold_delay <= 50
                line_s = (fit.param[2] >0) ? "-" : "--"
                line_c = (fit.param[2] >0) ? "b" : "c"
                axes[ax_idx].axvline(threshold_delay, ls=line_s, c=line_c, lineWidth=2)
            end
            axes[ax_idx].set_title("G: $graph_id | S: $(state_txt[s]) | dist:($(minimal_dist[1,s])/$(minimal_dist[2,s]))| EXP fit:$(round.(fit.param, digits=2)) | threshold $learning_threshold at $(round(threshold_delay, digits=1))", fontsize=10)
            axes[ax_idx].set_yticks((0.5, 1.0))
            axes[ax_idx].set_ylabel("correct", fontsize=10)
            axes[ax_idx].set_ylim([y_min_lim, 1.05])
        end
        axes[ax_idx].set_xlabel("n-th trajectory", fontsize=10)
        axes[ax_idx].set_xlim([0,35])
#         break
    end
        fig.subplots_adjust(hspace=1.1)
        fig.savefig(joinpath(save_path, "learning_curves_G$graph_id.png"), dpi = 600)
        fig.savefig(joinpath(save_path, "learning_curves_G$graph_id.pdf"), dpi = 600)
    return axes
end

function csv_state_decision_stats(graph_id::Int,
                            graph_state_counts;
                            learning_threshold=0.8,
                            min_visits_requirement = 2,
                            # filestr = "_Sim",
                            filestr = "",
                            # savepath = "./",
                            )

    savepath = makehomesavepath("learningcurves")
    mkdir(savepath)

    dfresults = []
    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    init_vals_lin = [0.5, 0.01]
    init_vals_exp = [0.5, 0.]
    dfresults_threshold_all = DataFrame()
    for s = states
        dfresults = DataFrame()
        println(s)
        counts_per_episode_actionwrong = vec(graph_state_counts[:,s,1])
        counts_per_episode_actioncorrect = vec(graph_state_counts[:,s,2])
        summed_counts_per_episode = counts_per_episode_actionwrong.+counts_per_episode_actioncorrect
        episodes_min_visits = findall(summed_counts_per_episode .>= min_visits_requirement)
        ratio = counts_per_episode_actioncorrect[episodes_min_visits] ./ summed_counts_per_episode[episodes_min_visits]
        weight = (summed_counts_per_episode[episodes_min_visits])
        nr_x = length(ratio)

        x_data = 1:nr_x
        y_data = ratio
        # markersize = 1+2*sqrt(weight[i]),
        threshold_delay = 0.
        weight = Float64.(weight)
        if minimal_dist[1,s] == minimal_dist[2,s]
            fit = curve_fit(model_lin, x_data, y_data, weight, init_vals_lin)
            y_hat = model_lin(x_data, fit.param)
            # axes[ax_idx].set_title("G: $graph_id | S: $(state_txt[s]) | dist:($(minimal_dist[1,s])/$(minimal_dist[2,s]))| LIN fit:$(round.(fit.param, digits=2))", fontsize=10)
        else
            fit = curve_fit(model_exp, x_data, y_data, weight, init_vals_exp) # Weighted cost function
            y_hat = model_exp(x_data, fit.param)
            threshold_delay = inv_model_exp(learning_threshold, fit.param)
            if threshold_delay > episodes_min_visits[end]
                threshold_delay = episodes_min_visits[end]
            end
            # axes[ax_idx].set_title("G: $graph_id | S: $(state_txt[s]) | dist:($(minimal_dist[1,s])/$(minimal_dist[2,s]))| EXP fit:$(round.(fit.param, digits=2)) | threshold $learning_threshold at $(round(threshold_delay, digits=1))", fontsize=10)
        end
        @show s
        @show threshold_delay
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_episodenumber")] = x_data
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_ratiocorrect")] = y_data
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_nrofsubjects")] = weight
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_fitteddata")] = y_hat
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_param1")] .= [fit.param[1]]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_param2")] .= [fit.param[2]]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_minimaldist")] .= ["$(minimal_dist[1,s])/$(minimal_dist[2,s])"]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay")] .= [threshold_delay]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_graphid")] .= [graph_id]

        CSV.write(joinpath(savepath, "learningcurve_G" * string(graph_id) * "_" * string(s) * "_" * state_txt[s] * filestr * ".csv"), dfresults,
            delim = " ");

        min_dist = sort(minimal_dist[:,s] .- 1.)
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay")] = [threshold_delay]
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_dist1vsdist2")] = [string(Int(min_dist[1]))*"-or-"*string(Int(min_dist[2]))]
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_graphid")] = [graph_id]

        CSV.write(joinpath(savepath, "learningcurvebar_true_G" * string(graph_id) * filestr * ".csv"), dfresults_threshold_all,
            delim = " ");
    end
end

# axes_1, deltaV_vs_Lerning1 = plot_state_decision_stats(1, graph_1_visits, state_txt_g1, graph_1_minimal_dist, transition_matrix_g1,Vvalues_g1)
# axes_3, deltaV_vs_Lerning3 = plot_state_decision_stats(3, graph_3_visits, state_txt_g3, graph_3_minimal_dist, transition_matrix_g3,Vvalues_g3)







# NOTE: NOT USED!!!
function csv_state_decision_stats_nofitting(graph_id::Int,
                            graph_state_counts;
                            learning_threshold=0.8,
                            min_visits_requirement = 2,
                            savepath = "/Users/vasia/Desktop/")
    dfresults = []
    graph_info = get_graph_info()
    states, minimal_dist, state_txt, graph = get_this_graph_info(graph_info, graph_id)

    dfresults_threshold_all = DataFrame()
    for s = states
        dfresults = DataFrame()
        println(s)
        counts_per_episode_actionwrong = vec(graph_state_counts[:,s,1])
        counts_per_episode_actioncorrect = vec(graph_state_counts[:,s,2])
        summed_counts_per_episode = counts_per_episode_actionwrong.+counts_per_episode_actioncorrect
        episodes_min_visits = findall(summed_counts_per_episode .>= min_visits_requirement)
        ratio = counts_per_episode_actioncorrect[episodes_min_visits] ./ summed_counts_per_episode[episodes_min_visits]
        weight = (summed_counts_per_episode[episodes_min_visits])
        nr_x = length(ratio)

        x_data = 1:nr_x
        y_data = ratio
        # markersize = 1+2*sqrt(weight[i]),
        threshold_delay = 0.

        if minimal_dist[1,s] != minimal_dist[2,s]
            threshold_delay = findfirst(x-> x.>=0.8, ratio)
        end
        @show threshold_delay
        if isnothing(threshold_delay)
            threshold_delay = Float64(episodes_min_visits[end])
        end
        @show s
        @show threshold_delay
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_episodenumber")] = x_data
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_ratiocorrect")] = y_data
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_nrofsubjects")] = weight
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_minimaldist")] .= ["$(minimal_dist[1,s])/$(minimal_dist[2,s])"]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay")] .= [threshold_delay]
        dfresults[!, Symbol(string(s)*"_"*state_txt[s]*"_graphid")] .= [graph_id]

        CSV.write(joinpath(savepath, "learningcurve_true_nofitting_G" * string(graph_id) * "_" * string(s) * "_" * state_txt[s] *".csv"), dfresults,
            delim = " ");

        min_dist = sort(minimal_dist[:,s] .- 1.)
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_thresholddelay")] = [threshold_delay]
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_dist1vsdist2")] = [string(Int(min_dist[1]))*"-or-"*string(Int(min_dist[2]))]
        dfresults_threshold_all[!, Symbol(string(s)*"_"*state_txt[s]*"_graphid")] = [graph_id]

        CSV.write(joinpath(savepath, "learningcurvebar_true_nofitting_G" * string(graph_id) *".csv"), dfresults_threshold_all,
            delim = " ");
    end
end
