# Markov Chain Monte Carlo (MCMC)
# Parallel Processing

using PyPlot
using Distributed
using Dates
using LinearAlgebra
using DelimitedFiles
using Random

struct Metropolis_Spec
    rl_impl::Function
    model_params::Array{MetropolisParam,1}
    nr_burnin::Int
    nr_samples::Int
    inter_collect_interval::Int
end

function getProposalDistribution(params)
    allVariances = [p.rwalk_step_size_std^2 for p in params]
    return Distributions.MvNormal(zeros(length(params)), diagm(allVariances))
end

function run_Metropolis(data::Array{Int,2}, mSpec, opts...)
    # Perform MCMC
    proposal_dist = getProposalDistribution(mSpec.model_params)
    states = zeros(mSpec.nr_samples, length(mSpec.model_params)+1)

    currentValues = [rand(p.prior_Distribution) for p in mSpec.model_params]

    index_alpha_0 = [i for i in 1:length(mSpec.model_params) if mSpec.model_params[i].name == "alpha_0"]
    if !isempty(index_alpha_0); index_alpha_0 = index_alpha_0[1]; end

    # ----- Make sure alpha_0 + alpha_surprise < 1
    # NOT USED eventually
    for i in 1:length(mSpec.model_params)
        if ((mSpec.model_params[i].name == "alpha_surprise")
            && !isempty(index_alpha_0) )
            while (currentValues[i] + currentValues[index_alpha_0]) > 1.
                currentValues[i] = rand(mSpec.model_params[i].prior_Distribution)
                currentValues[index_alpha_0] = rand(mSpec.model_params[index_alpha_0].prior_Distribution)
            end
        end
    end

    llPrior = sum([log(pdf(mSpec.model_params[i].prior_Distribution, currentValues[i])) for i in length(mSpec.model_params)])
    currentLL = llPrior + mSpec.rl_impl(data, currentValues, opts...)[1]

    nr_collected_samples = 0
    step_counter = 0
    while nr_collected_samples<mSpec.nr_samples
        if (step_counter > mSpec.nr_burnin && (step_counter%mSpec.inter_collect_interval ==0))
          nr_collected_samples += 1
          states[nr_collected_samples,:] = vcat(currentValues, currentLL)
        end
        proposalValues = currentValues + rand(proposal_dist)

        llPrior = 0.
        for i in 1:length(mSpec.model_params)
            if typeof(mSpec.model_params[i].prior_Distribution) == DiscreteUniform
                # Handle discrete variables as well)
                proposalValues[i] = Int(round(proposalValues[i]))
            end
            # ----- Make sure alpha_0 + alpha_surprise < 1
            # NOT USED eventually
            if ((mSpec.model_params[i].name == "alpha_surprise")
                && !isempty(index_alpha_0)
                && (proposalValues[i] > (1. - proposalValues[index_alpha_0])))
                llPrior += -Inf
            else
                llPrior += log(pdf(mSpec.model_params[i].prior_Distribution, proposalValues[i]))
            end
        end

        if llPrior > currentLL #prior can be -Inf for out of bound steps
            proposalLL = llPrior + mSpec.rl_impl(data, proposalValues, opts...)[1]
          if exp(proposalLL - currentLL) > rand()
              currentValues = proposalValues
              currentLL = proposalLL
          end
        else

        end

      step_counter+=1

    end
    println("llPrior=$llPrior")
    return states
end

function test_work_remotely(msg::String)
    println("worker proc id = $(myid()), called with echo message: $msg")
    RL_Fit.sayHi()
    a = rand(2, 3) .+ myid()
    return a
end

function run_parallel_Metropolis(data::Array{Int,2}, mSpec,rl_module_path::String,
                                opts...; nr_of_repetitions=2)
    # Run MCMC on many threads on server in parallel
    println("run_parallel_Metropolis")

    res_cat = @distributed vcat for i=1:nr_of_repetitions
        run_Metropolis(data, mSpec, opts...)
        end

    return res_cat
end

# specify CV for multiple models
struct Model_spec
    model_name::String
    learner_function_V::Function
    model_params::Array{MetropolisParam,1}
    result_folder::String
end


struct Model_spec_per_subj
    model_name::String
    learner_function_V::Function
    model_params::Array{MetropolisParam,1}
    per_param_posterior_std::Array{Float64,1}
    result_folder::String
end

# used to do a "batch" fit of multiple models
struct FitSpec_multiple_models
    all_models::Array{Any,1}  # allow  Metropolis_Spec or Metropolis_Spec_per_subj, TODO: learn inheritance in julia
    fit_max_episodes::Int
    nr_burnin::Int
    nr_samples::Int
    inter_collect_interval::Int
    top_n_fraction::Float64
    nr_repetitions::Int
    do_per_subject_fit::Bool
    log_path::String
end

function run_fit_multiple_models(description::String, data_path::String,
                                fit_spec::FitSpec_multiple_models, rl_module_path)
    fit_start_time = now()
    log_buffer = IOBuffer()
    log_print_ln!(log_buffer, "\nParameter fitting: start: $fit_start_time")
    log_print_ln!(log_buffer, "description: $description")
    plt.ioff() # don't plot, save figure

    (all_data, all_data_stats) = RL_Fit.load_SARSPEI(data_path; max_episodes_per_subj = fit_spec.fit_max_episodes)
    data_string = RL_Fit.to_short_string(all_data_stats)
    log_print_ln!(log_buffer, "Data:\n$data_string")
    nr_models = length(fit_spec.all_models)
    log_print_ln!(log_buffer, "nr of models: $nr_models")
    @show size(all_data)
    for i_model = 1:nr_models
        current_model = fit_spec.all_models[i_model]
        log_print_ln!(log_buffer, "********************************************************************************")
        log_print_ln!(log_buffer, "*********       Start parameter fitting of model $(current_model.model_name)       ************")
        log_print_ln!(log_buffer, "********************************************************************************")
        log_print_ln!(log_buffer, "Prior:")
        log_print_ln!(log_buffer, "$(current_model.model_params)")
        mSpec = RL_Fit.Metropolis_Spec(current_model.learner_function_V, current_model.model_params, fit_spec.nr_burnin, fit_spec.nr_samples, fit_spec.inter_collect_interval)
        # fit on all data (typically all subjects, max 40 episodes per subject)
        top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples,
            mcmc_samples, LL_noPrior_MAP, LL_noPrior_avgMAP, all_signals = RL_Fit.run_fit_procedure(description, all_data, mSpec,
                                                                    fit_spec.nr_repetitions,  "TODO: mcmc_data_file",
                                                                    fit_spec.top_n_fraction, rl_module_path)
        log_print_ln!(log_buffer, "top_MAP: $top_MAP")
        log_print_ln!(log_buffer, "top_n_avgMAP: $top_n_avgMAP")
        log_print_ln!(log_buffer, "size(mcmc_samples): $(size(mcmc_samples))")
        log_print_ln!(log_buffer, "LL(top_MAP): $LL_noPrior_MAP")
        log_print_ln!(log_buffer, "LL(top_n_avgMAP): $LL_noPrior_avgMAP")

        top_n_diff = (top_n_max - top_n_min)[1:end-1]
        if maximum(top_n_diff) > 0.15
            log_print_ln!(log_buffer, "   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            log_print_ln!(log_buffer, "       !! CHECK LARGE DIFFERENCE IN TOP-N PARAM VALUES !!")
            log_print_ln!(log_buffer, "       top_n_diff=$top_n_diff")
            log_print_ln!(log_buffer, "       top_n_min: $top_n_min")
            log_print_ln!(log_buffer, "       top_n_max: $top_n_max")
            log_print_ln!(log_buffer, "   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        end

        nr_params = length(current_model.model_params)
        nr_subjects = length(all_data_stats.all_subject_ids)
        AIC = 2.0 * nr_params - 2.0 * LL_noPrior_MAP
        AICc = AIC + (2*nr_params)*(nr_params+1)/(nr_subjects- nr_params - 1)

        # Store the map parameters and the log likelihood
        LL_path = "$(current_model.result_folder)maxLL_fit/"
        if !ispath(LL_path)
            println("mkpath($LL_path)")
            mkpath("$LL_path")
        end
        writedlm("$(LL_path)LogLikelihood_MAP.csv", LL_noPrior_MAP)
        writedlm("$(LL_path)LogLikelihood_avgMAP.csv", LL_noPrior_avgMAP)
        writedlm("$(LL_path)MAP_params.csv", top_MAP[1:end-1])
        writedlm("$(LL_path)MAP_n_avg_params.csv", top_n_avgMAP[1:end-1])
        writedlm("$(LL_path)AIC.csv", AIC)
        writedlm("$(LL_path)AICc.csv", AICc)
        writedlm("$(LL_path)mcmc_samples.csv", mcmc_samples)
        writedlm("$(LL_path)var_covar_mcmc_samples.csv", cov(mcmc_samples))

        # This stores the results and signals of the 1st subject:
        # open("$(LL_path)step_by_step_info.csv", "w") do newfile
        # write(newfile, to_string(all_signals[1]))
        # end

        # Plot: re-call get_MAP()
        f = RL_Fit.plot_result(mcmc_samples, current_model.model_params,
            "$(current_model.model_name), $data_string, LL=$(round(LL_noPrior_MAP,digits=2)) | AIC=$(round(AIC,digits=2))",
            top_n_fraction = 1.)
        f.savefig("$(LL_path)MCMC.png")   # save the figure to file
        plt.close("all")

        f = RL_Fit.plot_result_logLL(mcmc_samples, current_model.model_params,
            "$(current_model.model_name), $data_string, LL=$(round(LL_noPrior_MAP,digits=2)) | AIC=$(round(AIC,digits=2))",
            top_n_fraction = 1.)
        f.savefig("$(LL_path)MCMC_LLdots.png")   # save the figure to file
        plt.close("all")

        f = RL_Fit.plot_result(mcmc_samples, current_model.model_params,
            "$(current_model.model_name), $data_string, LL=$(round(LL_noPrior_MAP,digits=2)) | AIC=$(round(AIC,digits=2))",
            top_n_fraction = 1.,
            hist_density = false)
        f.savefig("$(LL_path)MCMC_notnormed.png")   # save the figure to file
        plt.close("all")

        if fit_spec.do_per_subject_fit
            log_print_ln!(log_buffer, "Per subject fit:")

            all_parameters_subj = []
            params_with_pop_prior = []
            sum_pop_LL = 0.
            sum_subj_LL = 0.
            which_top = deepcopy(top_MAP)
            for i in 1:length(current_model.model_params)
                current_param = current_model.model_params[i]
                model_param_distr = current_param.prior_Distribution

                prior_mean = which_top[i]
                prior_std = current_model.per_param_posterior_std[i]
                prior_param_distr = TruncatedNormal(prior_mean, prior_std,
                                                    minimum(model_param_distr),
                                                    maximum(model_param_distr))

                model_param_prior = MetropolisParam(
                                    current_param.name,
                                    current_param.descr,
                                    prior_param_distr,
                                    current_param.rwalk_step_size_std)

                push!(params_with_pop_prior, model_param_prior)
            end

            for s_id in all_data_stats.all_subject_ids
              log_print!(log_buffer, "fitting subject_id $s_id. ")

              subj_signals, pop_signals = RL_Fit.run_single_subj_fit_write_csv(
                                        data_path,
                                        "$(LL_path)",
                                        s_id,
                                        fit_spec.fit_max_episodes,
                                        current_model.learner_function_V,
                                        params_with_pop_prior,
                                        which_top[1:end-1],
                                        fit_spec.nr_burnin,
                                        fit_spec.nr_samples,
                                        fit_spec.inter_collect_interval,
                                        fit_spec.nr_repetitions,
                                        fit_spec.top_n_fraction,
                                        description, rl_module_path)

              push!(all_parameters_subj,subj_signals.param_vector)
              log_print!(log_buffer, "subj_signals.param_vector= $(round.(subj_signals.param_vector,digits=3))")
              log_print_ln!(log_buffer, "subj_LL= $(round(subj_signals.LL,digits=3)), pop_LL= $(round(pop_signals.LL,digits=3))")
              sum_pop_LL += pop_signals.LL
              sum_subj_LL += subj_signals.LL
            end
            log_print_ln!(log_buffer, "sum_pop_LL= $(round(sum_pop_LL,digits=3)), sum_subj_LL=$(round(sum_subj_LL,digits=3))")
            # write results
            RL_Fit.writedlm("$(LL_path)all_subj_param_vectors.csv", all_parameters_subj)
            RL_Fit.writedlm("$(LL_path)sum_pop_param_LL.csv", sum_pop_LL)
            RL_Fit.writedlm("$(LL_path)sum_subj_param_LL.csv", sum_subj_LL)

        end
        log_print_ln!(log_buffer, "END fitting of model $(current_model.model_name)")
        log_print_ln!(log_buffer, "================================================================================")
    end
    log_print_ln!(log_buffer, "END: $(now())")
    # write log file
     log_file_name = "$(fit_spec.log_path)/Model_fit_log_$(Dates.format(now(), "dd_u_yyyy_HH_MM")).txt"
     log_str = String(take!(log_buffer))
     open(log_file_name, "w") do f
         write(f, log_str)
     end
end
