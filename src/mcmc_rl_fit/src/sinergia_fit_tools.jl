# high level functions used in the project
using PyPlot
using DelimitedFiles


function run_fit_procedure(description::String, data::Array{Int,2}, algo_fit_spec,
                            nr_repetitions::Int, mcmc_data_file::String,
                            top_n_fraction::Float64, rl_module_path::String, opts...)
    start_time = now()
    println("start: $start_time")
    println(description)
    data_string = RL_Fit.to_short_string(RL_Fit.get_data_stats(data))
    println(data_string)
        flush(stdout)
    println("run_fit_procedure")

    mcmc_samples = RL_Fit.run_parallel_Metropolis(data, algo_fit_spec, rl_module_path, opts...; nr_of_repetitions=nr_repetitions)
    nr_map_samples = Int(ceil(size(mcmc_samples)[1] * top_n_fraction))
    println("sampling data size: $(size(mcmc_samples)), top_N: $nr_map_samples")
    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples = RL_Fit.get_MAP(mcmc_samples, nr_map_samples)
    println("result timestamp: $(now())")
    println("top_MAP: $(round.(top_MAP,digits=4)) (Max Likelihood estimate (top 1 sample). LL includes Prior LL)")
    println("top_n_avgMAP: $(round.(top_n_avgMAP,digits=4)) . (Avg over top $nr_map_samples samples. LL includes Prior LL)")
    println("top_n_min: $(round.(top_n_min,digits=4))")
    println("top_n_max: $(round.(top_n_max,digits=4))")
    # evaluate using top_n_avgMAP parameter. get the LL without the prior (penalty)
    LL_noPrior_avgMAP, all_signals_avg = algo_fit_spec.rl_impl(data, top_n_avgMAP[1:end-1], opts...)
    println("LL top_n_avgMAP=$(round(LL_noPrior_avgMAP, digits=4))")
    # evaluate using top_MAP parameter. get the LL without the prior (penalty)
    LL_noPrior_MAP, all_signals = algo_fit_spec.rl_impl(data, top_MAP[1:end-1], opts...)
    println("LL top_MAP=$(round(LL_noPrior_MAP, digits=4))")

    flush(stdout)
    end_time = now()
    println("end of fitting: $end_time, duration: $(end_time-start_time)")
    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples, mcmc_samples, LL_noPrior_MAP, LL_noPrior_avgMAP, all_signals
end


function _write_subj_data_file(subj_data_path::String , data_stats::Data_stats ,subj_signals::Step_by_step_signals, pop_signals::Step_by_step_signals, subj_fit_mcmc_figure)
    # writes the data to the file system.
    # the directory subj_data_path is created if it does not yet exist.
    if !ispath(subj_data_path)
        mkpath(subj_data_path)
    end

    #learning signals when per-subject (posterior) params are used
    path = "$(subj_data_path)params_subj.csv"
    writedlm(path, subj_signals.param_vector)

    # always write the action_selection_prob
    writedlm("$(subj_data_path)action_selection_prob_subj.csv", subj_signals.action_selection_prob)

    if subj_fit_mcmc_figure != nothing
    # keep track of the MCMC samples for later visualizations
    # writedlm("$(subj_data_path)/mcmc_samples_subj.csv", subj_fit_mcmc_samples)
    # plot
        subj_fit_mcmc_figure.savefig("$(subj_data_path)MCMC.png")   # save the figure to file
    end

    if length(subj_signals.RPE)>0
      path = "$(subj_data_path)RPE_subj.csv"
        d = hcat(subj_signals.RPE, subj_signals.action_selection_prob)
        writedlm(path, d)
    end
    if length(subj_signals.SPE)>0
        path = "$(subj_data_path)SPE_subj.csv"
        d = hcat(subj_signals.SPE, subj_signals.action_selection_prob)
        writedlm(path, d)
    end

    # learning signals when population (prior) params are used:
    path = "$(subj_data_path)params_pop.csv"
    writedlm(path, pop_signals.param_vector)

    # always write the action_selection_prob
    writedlm("$(subj_data_path)action_selection_prob_pop.csv", pop_signals.action_selection_prob)

    if length(pop_signals.RPE)>0
        path = "$(subj_data_path)RPE_pop.csv"
          d = hcat(pop_signals.RPE, pop_signals.action_selection_prob)
          writedlm(path, d)

    end
    if length(pop_signals.SPE)>0
        path = "$(subj_data_path)SPE_pop.csv"
        writedlm(path, hcat(pop_signals.SPE, pop_signals.action_selection_prob))
    end

    if haskey(pop_signals.results_dict, "RPE_QV")
        path = "$(subj_data_path)RPE_QV_pop.csv"
        writedlm(path, pop_signals.results_dict["RPE_QV"])
    end
    if haskey(subj_signals.results_dict, "RPE_QV")
        path = "$(subj_data_path)RPE_QV_subj.csv"
        writedlm(path, subj_signals.results_dict["RPE_QV"])
    end

    if haskey(pop_signals.results_dict, "trialByTrial_Vs")
        path = "$(subj_data_path)StateValue_pop.csv"
        writedlm(path, pop_signals.results_dict["trialByTrial_Vs"])
    end
    if haskey(subj_signals.results_dict, "trialByTrial_Vs")
        path = "$(subj_data_path)StateValue_subj.csv"
        writedlm(path, subj_signals.results_dict["trialByTrial_Vs"])
    end

    ###############################
    # Store V-values (if there are any in the results_dict)
    if haskey(pop_signals.results_dict, "StateValue_V1_record")
        path = "$(subj_data_path)StateValue_V1_pop.csv"
        writedlm(path, pop_signals.results_dict["StateValue_V1_record"])
    end
    if haskey(subj_signals.results_dict, "StateValue_V1_record")
        path = "$(subj_data_path)StateValue_V1_subj.csv"
        writedlm(path, subj_signals.results_dict["StateValue_V1_record"])
    end
    if haskey(pop_signals.results_dict, "StateValue_V2_record")
        path = "$(subj_data_path)StateValue_V2_pop.csv"
        writedlm(path, pop_signals.results_dict["StateValue_V2_record"])
    end
    if haskey(subj_signals.results_dict, "StateValue_V2_record")
        path = "$(subj_data_path)StateValue_V2_subj.csv"
        writedlm(path, subj_signals.results_dict["StateValue_V2_record"])
    end
    if haskey(pop_signals.results_dict, "StateValue_V3_record")
        path = "$(subj_data_path)StateValue_V3_pop.csv"
        writedlm(path, pop_signals.results_dict["StateValue_V3_record"])
    end
    if haskey(subj_signals.results_dict, "StateValue_V3_record")
        path = "$(subj_data_path)StateValue_V3_subj.csv"
        writedlm(path, subj_signals.results_dict["StateValue_V3_record"])
    end
    ##############################

    if haskey(pop_signals.results_dict, "Surprise")
        path = "$(subj_data_path)Surprise_pop.csv"
        writedlm(path, pop_signals.results_dict["Surprise"])
    end
    if haskey(subj_signals.results_dict, "Surprise")
        path = "$(subj_data_path)Surprise_subj.csv"
        writedlm(path, subj_signals.results_dict["Surprise"])
    end
    if haskey(pop_signals.results_dict, "gamma_surprise")
        path = "$(subj_data_path)gamma_surprise_pop.csv"
        writedlm(path, pop_signals.results_dict["gamma_surprise"])
    end
    if haskey(subj_signals.results_dict, "gamma_surprise")
        path = "$(subj_data_path)gamma_surprise_subj.csv"
        writedlm(path, subj_signals.results_dict["gamma_surprise"])
    end
    ##############################
    if haskey(pop_signals.results_dict, "DeltaPolicy")
        path = "$(subj_data_path)DeltaPolicy_pop.csv"
        writedlm(path, pop_signals.results_dict["DeltaPolicy"])
    end
    if haskey(subj_signals.results_dict, "DeltaPolicy")
        path = "$(subj_data_path)DeltaPolicy_subj.csv"
        writedlm(path, subj_signals.results_dict["DeltaPolicy"])
    end
    #
    if haskey(pop_signals.results_dict, "PolicyParametersImbalance")
        path = "$(subj_data_path)PolicyParametersImbalance_pop.csv"
        writedlm(path, pop_signals.results_dict["PolicyParametersImbalance"])
    end
    if haskey(subj_signals.results_dict, "PolicyParametersImbalance")
        path = "$(subj_data_path)PolicyParametersImbalance_subj.csv"
        writedlm(path, subj_signals.results_dict["PolicyParametersImbalance"])
    end
    #
    if haskey(pop_signals.results_dict, "PolicyImbalance")
        path = "$(subj_data_path)PolicyImbalance_pop.csv"
        writedlm(path, pop_signals.results_dict["PolicyImbalance"])
    end
    if haskey(subj_signals.results_dict, "PolicyImbalance")
        path = "$(subj_data_path)PolicyImbalance_subj.csv"
        writedlm(path, subj_signals.results_dict["PolicyImbalance"])
    end
    #
    if haskey(pop_signals.results_dict, "PolicyParametersImbalanceAllstates")
        path = "$(subj_data_path)PolicyParametersImbalanceAllstates_pop.csv"
        writedlm(path, pop_signals.results_dict["PolicyParametersImbalanceAllstates"])
    end
    if haskey(subj_signals.results_dict, "PolicyParametersImbalanceAllstates")
        path = "$(subj_data_path)PolicyParametersImbalanceAllstates_subj.csv"
        writedlm(path, subj_signals.results_dict["PolicyParametersImbalanceAllstates"])
    end
    #
    if haskey(pop_signals.results_dict, "trial_by_trial_p_sa_max")
        path = "$(subj_data_path)action_selection_prob_MAX_pop.csv"
        writedlm(path, pop_signals.results_dict["trial_by_trial_p_sa_max"])
    end
    if haskey(subj_signals.results_dict, "trial_by_trial_p_sa_max")
        path = "$(subj_data_path)action_selection_prob_MAX_subj.csv"
        writedlm(path, subj_signals.results_dict["trial_by_trial_p_sa_max"])
    end

    # LL of subject
    path = "$(subj_data_path)LL_pop.csv"
    writedlm(path, pop_signals.LL)

    path = "$(subj_data_path)LL_subj.csv"
    writedlm(path, subj_signals.LL)

    # BIC of subject
    path = "$(subj_data_path)BIC_pop.csv"
    BIC_pop = get_BIC(pop_signals.LL, length(pop_signals.param_vector), data_stats.nr_actions)
    writedlm(path, BIC_pop)

    path = "$(subj_data_path)BIC_subj.csv"
    BIC_subj = get_BIC(subj_signals.LL, length(subj_signals.param_vector), data_stats.nr_actions)
    writedlm(path, BIC_subj)

    #
    if haskey(pop_signals.results_dict, "PolicyParametersDiffChosenVsUnchosen")
        path = "$(subj_data_path)PolicyParametersDiffChosenVsUnchosen_pop.csv"
        writedlm(path, pop_signals.results_dict["PolicyParametersDiffChosenVsUnchosen"])
    end
    if haskey(subj_signals.results_dict, "PolicyParametersDiffChosenVsUnchosen")
        path = "$(subj_data_path)PolicyParametersDiffChosenVsUnchosen_subj.csv"
        writedlm(path, subj_signals.results_dict["PolicyParametersDiffChosenVsUnchosen"])
    end
    ##############################

    # textfile describing unstructured data for info only
    path = "$(subj_data_path)description.txt"
    f = open(path, "w")
    write(f,"DateTime of fitting: $(now())\n")
    write(f,"DATA_STATS\n")
    write(f,to_string(data_stats))
    write(f,"\nSUBJECT_SIGNALS\n")
    write(f, to_string(subj_signals))
    write(f,"\nPOPULATION_SIGNALS\n")
    write(f, to_string(pop_signals))
    close(f)
end


function run_single_subj_fit_write_csv(
    data_path::String,
    base_path::String,
    id_subject::Int64,
    fit_max_episodes_per_subj::Int64,
    learner_function_V::Function,
    population_prior,
    population_map_params,
    burn_in::Int64,
    nr_mcmc_samples::Int64,
    inter_collect_interval::Int64,
    nr_of_repetitions::Int64,
    top_n_fraction::Float64,
    descritpion::String,
    rl_module_path::String;
    write_mcmc_picture_file = true)
    # a higher level script that writes all the information we currently
    # want to the file system.

    data_single_subj, data_stats = load_SARSPEI(data_path; subject_id = id_subject,
                                            max_episodes_per_subj=fit_max_episodes_per_subj)

    mSpec = Metropolis_Spec(learner_function_V, population_prior, burn_in,nr_mcmc_samples, inter_collect_interval)
    mcmc_samples = run_parallel_Metropolis(data_single_subj, mSpec, rl_module_path; nr_of_repetitions=nr_of_repetitions)

    nr_map_samples = Int(ceil(size(mcmc_samples)[1] * top_n_fraction))
    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples = get_MAP(mcmc_samples, nr_map_samples)

    map_val_subj = deepcopy(top_MAP)

    println("subj: $id_subject, population prior, map_val_subj: $(round.(map_val_subj,digits=4))")
    flush(stdout)
    # for evaluation (unlike fitting) load ALL Episodes
    data_single_subj, data_stats = load_SARSPEI(data_path; subject_id = id_subject)
    LL, subj_signals = learner_function_V(data_single_subj, map_val_subj)
    subj_signals = subj_signals[id_subject]

    LL_pop, pop_signals = learner_function_V(data_single_subj, population_map_params)
    pop_signals = pop_signals[id_subject]

    subj_fit_mcmc_figure = nothing
    if write_mcmc_picture_file
        fig_title = "Subj:$id_subject, $(RL_Fit.to_short_string(data_stats)), $learner_function_V, LL_sub: $(round(LL,digits=2)), LL_pop: $(round(LL_pop,digits=2))"
        subj_fit_mcmc_figure = RL_Fit.plot_result(mcmc_samples, population_prior, fig_title)#, top_n_fraction = 1.)
    end

    subj_data_path = "$(base_path)subj_$(id_subject)/"
    _write_subj_data_file(subj_data_path, data_stats,subj_signals, pop_signals, subj_fit_mcmc_figure)
    plt.close("all")
    subj_signals, pop_signals, mcmc_samples, data_stats
end


function get_BIC(LL, nr_params, nr_actions)
    BIC = -2. * LL + nr_params * log(nr_actions)
end

function get_BIClogevidence(LL, nr_params, nr_actions)
    BIClogevidence = LL - nr_params * log(nr_actions) / 2.
    # or
    # BIClogevidence = - get_BIC(LL, nr_params, nr_actions) / 2.
end
