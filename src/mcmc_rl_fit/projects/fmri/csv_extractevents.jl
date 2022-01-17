# Extract some info from participants' data
# (i.e SARSPEICZVG_all_fMRI.csv)
using DelimitedFiles

function csv_extractallevents()
    # Process participants' data and create the files that are now in
    # /src/mcmc_rl_fit/projects/fmri/someresults/fits/all_events
    
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"

    base_path = makehomesavepath("all_events")
    mkdir(base_path)

    all_data, all_stats = RL_Fit.load_SARSPEI(data_path)

    all_subject_ids = all_stats.all_subject_ids

    for subj_id in all_subject_ids

        subj_data = RL_Fit.filter_by_person(all_data, subj_id)

        subj_path = joinpath(base_path, "subj_"*string(subj_id))

        mkdir(subj_path)
        # mkpath(subj_path)
        writedlm(subj_path*"/states.csv", subj_data[:, Int(RL_Fit.c_state)])
        writedlm(subj_path*"/actions.csv", subj_data[:, Int(RL_Fit.c_action)])
        writedlm(subj_path*"/rewards.csv", subj_data[:, Int(RL_Fit.c_reward)])
        writedlm(subj_path*"/is_surprise_trial.csv", subj_data[:, Int(RL_Fit.c_surprise_trial)])
        writedlm(subj_path*"/graph_version.csv", subj_data[1, Int(RL_Fit.c_graph_version)])

    end

end

# NOTE: the nr_states are actually the number of actions
#(ie excluding start states)
# And then, the % of surprise trials is the nr_surprise_trials/nr_states
# (ie number of actions = transitions).
function getnrofevents()
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"

    all_data, all_stats = RL_Fit.load_SARSPEI(data_path)

    all_subject_ids = all_stats.all_subject_ids

    nr_states = []
    nr_episodes = []
    nr_surprise_trials = []
    for subj_id in all_subject_ids

        subj_data = RL_Fit.filter_by_person(all_data, subj_id)

        push!(nr_states, length(subj_data[:, Int(RL_Fit.c_state)]))
        push!(nr_episodes, length(findall(subj_data[:, Int(RL_Fit.c_reward)] .== 1)))
        push!(nr_surprise_trials, length(findall(subj_data[:, Int(RL_Fit.c_surprise_trial)] .== 1)))
    end
    nr_states, nr_episodes, nr_surprise_trials
end
