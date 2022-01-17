using PyPlot
using CSV, DataFrames, DataFramesMeta
using HypothesisTests

""" For Fig.2A
Compute average path length across real or simulated (edit!!) participants
"""
function get_pathlength(;
                        filestr = "",
                        # filestr = "_Sim",
                        # savepath = "./",
                        )
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ------ Which data? Real or simulated? Edit!

    # Real participants' data:
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEIC_all_fMRI.csv"

    # Simulated data:
    # data_path = "./mcmc_rl_fit/projects/fmri/data_sim/SARSPEICZVG_all_fMRI_Sim_exporder_SurpAC_run2.csv"

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    save_path = makehomesavepath("pathlength")
    mkdir(save_path)

    max_episodes_per_subj = 60
    (all_data, all_stats) = RL_Fit.load_SARSPEI(data_path;max_episodes_per_subj=max_episodes_per_subj)
    all_subject_ids = all_stats.all_subject_ids
    println("all_subject_ids: $all_subject_ids")
    nr_of_subjects = length(all_subject_ids)

    pathlength_all_subjects = zeros(nr_of_subjects, max_episodes_per_subj)
    row_counter=1

    for subj_id in all_subject_ids
        subj_data = RL_Fit.filter_by_person(all_data,subj_id)
        idx_reward = findall(subj_data[:,Int(RL_Fit.c_reward)].>0)
        pathlength = diff(vcat(0,idx_reward))
        nr_episodes = length(pathlength)
        pathlength_all_subjects[row_counter, 1:nr_episodes] = pathlength
        pathlength_all_subjects[row_counter, nr_episodes+1:end] .= NaN
        #     figure()
        #     plot(pathlength)
        row_counter+=1

    end
    m, s, serror = calc_stats_nan(pathlength_all_subjects')
    weights = nanlength(pathlength_all_subjects',2)

    dfresults = DataFrame()
    dfresults[!, Symbol("meanlength")] = m
    dfresults[!, Symbol("stdlength")] = s
    dfresults[!, Symbol("selength")] = serror
    dfresults[!, Symbol("weights")] = weights[:]

    CSV.write(joinpath(save_path, "meanpathlength"*filestr*".csv"), dfresults,
        delim = " ");

    dfresults_perepisode = DataFrame()
    for i_ep in 1:max_episodes_per_subj
        dfresults_perepisode[!, Symbol("ep_"*string(i_ep))] = pathlength_all_subjects[:,i_ep]
    end
    CSV.write(joinpath(save_path, "pathlengthperepisode"*filestr*".csv"), dfresults_perepisode,
        delim = " ");

    # For random search comparison:
    # pathlength_firstepisode = pathlength_all_subjects[:,1]
    # pathlength_firstepisode_minusrandom = pathlength_firstepisode .- 6.
    #
    # @show OneSampleTTest(pathlength_firstepisode_minusrandom)
    # pval = pvalue(OneSampleTTest(pathlength_firstepisode_minusrandom))
    # @show pval

    pathlength_all_subjects, m, s, serror
end
