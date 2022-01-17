# Cross Validation procedures
using Dates
using LinearAlgebra
using DelimitedFiles
using Random

struct OneFoldSpec
    in_sample_ids::Set
    out_of_sample_ids::Set
end

struct CV_multiple_models_spec
    all_models::Any
    folds_spec::Array{OneFoldSpec, 1}
    nr_burnin::Int
    nr_samples::Int
    inter_collect_interval::Int
    top_n_fraction::Float64
    nr_repetitions::Int
    log_path::String
end

struct CV_result
    model::Any  # the algorith that is used to produce this result
    in_sample_LL_list::Array{Float64,2}
    out_of_sample_LL_list::Array{Float64,2}
    per_fold_avgMAPs::Array{Float64,2}
    per_fold_MAPs::Array{Float64,2}
    start_time::DateTime
    end_time::DateTime
    duration::Dates.Millisecond
    LL_out_subjectwise::Array{Float64,2}
end

struct CV_multiple_models_result
    all_results::Array{CV_result,1}
    start_time::DateTime
    end_time::DateTime
    duration::Dates.Millisecond
end


function get_k_folds(all_subj_ids::Set, out_of_sample_subj_sets::Array)
    k_folds = length(out_of_sample_subj_sets)
    folds_spec = Array{OneFoldSpec, 1}(undef, k_folds)
    println(typeof(folds_spec))
    for k in 1:k_folds
        in_sample = copy(all_subj_ids)
        out_of_sample = out_of_sample_subj_sets[k]
        for i in out_of_sample delete!(in_sample, i) end
        out_of_sample = Set(out_of_sample)
        oneFspec = OneFoldSpec(in_sample,out_of_sample)
        folds_spec[k] = oneFspec
    end
    folds_spec
end

function get_k_folds_all_out(all_subj_ids::Set, k_folds::Int)
    nr_subj = length(all_subj_ids)
    if k_folds > nr_subj
        error("Error in get_k_folds_all_out: k_folds = $k_folds > nr_subj=$nr_subj")
    end
    if k_folds <= 1
        error("Error in get_k_folds_all_out: k_folds = $k_folds <=1")
    end

    out_of_sample_sets = Array{Set, 1}(undef, k_folds)
    for k = 1:k_folds
        out_of_sample_sets[k] = Set()
    end

    s1=randperm(nr_subj)
    out_of_sample_subj_ids = collect(all_subj_ids)[s1]
    println(out_of_sample_subj_ids)
    for i = 1:nr_subj
        i_fold = 1+((i-1) % k_folds)
        push!(out_of_sample_sets[i_fold], out_of_sample_subj_ids[i])
    end
    get_k_folds(all_subj_ids,  out_of_sample_sets)
end

function get_k_folds(all_subj_ids::Set, k_folds::Int, out_of_sample_N::Int)
    # produces k folds, each having the same number (out_of_sample_N ) of subjects
    # in the out of sample set. Note: It is possible that not ALL subjects are in
    # one of the out-of-sample sets.
    # Use get_k_folds_all_out to get a specification which has ALL subjects out
    # of sample. In that case, not all sets are of the same size.
    nr_subj = length(all_subj_ids)
    max_i = k_folds * out_of_sample_N
    if max_i > nr_subj
        error("Error in get_k_folds: k_folds * out_of_sample_N = $max_i > nr_subj=$nr_subj")
    end
    s1=randperm(nr_subj)[1:max_i]
    out_of_sample_subj_ids = collect(all_subj_ids)[s1]
    out_of_sample_subj_pairs = reshape(out_of_sample_subj_ids,(k_folds,out_of_sample_N))
    # println(out_of_sample_subj_pairs)
    out_of_sample_sets = Array{Set, 1}(undef, k_folds)
    for k = 1:k_folds
        out_of_sample_sets[k] = Set(out_of_sample_subj_pairs[k,:])
    end
    get_k_folds(all_subj_ids, k_folds, out_of_sample_sets)
end

function run_cross_validation(description::String, data::Array{Int,2},
                        cv_spec::CV_multiple_models_spec, rl_module_path::String)
    cv_start_time = now()
    log_buffer = IOBuffer()
    log_print_ln!(log_buffer, "\nCROSS VALIDATION: start: $cv_start_time")
    log_print_ln!(log_buffer, "description: $description")
    data_string = RL_Fit.to_short_string(RL_Fit.get_data_stats(data))
    log_print_ln!(log_buffer, "Data:\n$data_string")
    nr_models = length(cv_spec.all_models)
    log_print_ln!(log_buffer, "nr of models: $nr_models")

    for i_model = 1:nr_models
        current_model = cv_spec.all_models[i_model]
        log_print_ln!(log_buffer, "********************************************************************************")
        log_print_ln!(log_buffer, "*********       Start cross validation of model $(current_model.model_name)       ************")
        log_print_ln!(log_buffer, "********************************************************************************")
        cv_result_model = run_CV_single_model(
            data,
            current_model,
            cv_spec.folds_spec,
            cv_spec.nr_burnin,
            cv_spec.nr_samples,
            cv_spec.inter_collect_interval,
            cv_spec.top_n_fraction,
            cv_spec.nr_repetitions,
            log_buffer,
            rl_module_path
        )
        write_CV_result(cv_result_model, cv_spec.folds_spec)
        log_print_ln!(log_buffer, "summed out-of-sample LL: $(sum(cv_result_model.out_of_sample_LL_list))")
        log_print_ln!(log_buffer, "END cross validation of model $(current_model.model_name), duration: $(cv_result_model.duration)")
        log_print_ln!(log_buffer, "================================================================================")
    end

     log_file_name = "$(cv_spec.log_path)/CV_log_$(Dates.format(now(), "dd_u_yyyy_HH_MM")).txt"

     log_str = String(take!(log_buffer))
     open(log_file_name, "w") do f
         write(f, log_str)
     end
end

function run_CV_single_model(data::Array{Int,2}, model_spec::Any,
                            folds_spec::Array{OneFoldSpec,1},
                            nr_burnin::Int,
                            nr_samples::Int,
                            inter_collect_interval::Int,
                            top_n_fraction::Float64,
                            nr_repetitions::Int,
                            log_buffer::IOBuffer,
                            rl_module_path::String
                            )
    start_time=now()
    nr_of_folds = length(folds_spec)
    nr_of_params = length(model_spec.model_params)
    in_sample_LL_list = zeros(Float64,nr_of_folds, 2)
    out_of_sample_LL_list = zeros(Float64, nr_of_folds, 2)
    per_fold_avgMAPs = Array{Float64, 2}(undef, (nr_of_folds, nr_of_params+2) )  # +2: space for LL and NR of subj
    per_fold_MAPs = Array{Float64, 2}(undef, (nr_of_folds, nr_of_params+2) )  # +2: space for LL and NR of subj

    allSubj = sort(unique(data[:, Int(c_subject)]))
    nr_subjects = length(allSubj)
    LL_out_subjectwise = zeros(Float64, nr_of_folds, nr_subjects)

    for k in 1:nr_of_folds
       log_print_ln!(log_buffer, "   START FOLD #$k")
       in_sample_data = RL_Fit.filter_by_person(data, folds_spec[k].in_sample_ids)
       out_of_sample_data = RL_Fit.filter_by_person(data, folds_spec[k].out_of_sample_ids)
        log_print_ln!(log_buffer, "in-sample_subjects: $(folds_spec[k].in_sample_ids)")
        log_print_ln!(log_buffer, "out-of-sample_subjects: $(folds_spec[k].out_of_sample_ids)")
        mcmc_spec = Metropolis_Spec(
            model_spec.learner_function_V,
            model_spec.model_params,
            nr_burnin,
            nr_samples,
            inter_collect_interval)

        mcmc_samples = RL_Fit.run_parallel_Metropolis(in_sample_data, mcmc_spec, rl_module_path; nr_of_repetitions=nr_repetitions)
        nr_map_samples = Int(ceil(size(mcmc_samples)[1] * top_n_fraction))
        log_print_ln!(log_buffer, "sampling data size: $(size(mcmc_samples)), top_N: $nr_map_samples")
        top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples = RL_Fit.get_MAP(mcmc_samples, nr_map_samples)

        in_sample_params = top_MAP[1:end-1]
        per_fold_avgMAPs[k,:] = hcat(top_n_avgMAP, length(folds_spec[k].in_sample_ids))

        top_MAP = reshape(top_MAP, 1, :) # Convert vector (nparam,) to array (nparam x 1)
        per_fold_MAPs[k,:] = hcat(top_MAP, length(folds_spec[k].in_sample_ids))

        LL_in = model_spec.learner_function_V(in_sample_data, in_sample_params)[1]

        in_sample_LL_list[k,:] = [LL_in length(folds_spec[k].in_sample_ids)]

        # get out of sample LL
        LL_out = model_spec.learner_function_V(out_of_sample_data, in_sample_params)[1]
        out_of_sample_LL_list[k,:] = [LL_out  length(folds_spec[k].out_of_sample_ids)]

        # Save the LL of each of the out-of-sample subjects (to be used for Bayesian Model Selection)
        for out_subj in folds_spec[k].out_of_sample_ids
            out_of_sample_data_subject = RL_Fit.filter_by_person(data, out_subj)
            LL_out_subjectwise[k, out_subj] = model_spec.learner_function_V(out_of_sample_data_subject, in_sample_params)[1]
        end

        log_print_ln!(log_buffer, "fold #$k result: top_MAP=$(round.(top_MAP, digits=3)), out of sample LL=$(round(LL_out, digits=3))")
        log_print_ln!(log_buffer, "END of fold #$k. timestamp: $(now())")
    end
    end_time = now()
    results =  CV_result(
        model_spec,
        in_sample_LL_list,
        out_of_sample_LL_list,
        per_fold_avgMAPs,
        per_fold_MAPs,
        start_time,
        end_time,
        end_time-start_time,
        LL_out_subjectwise)
end

function write_CV_result(cv_res::CV_result, f_spec)
    println("Write results to file")

    cv_path = "$(cv_res.model.result_folder)/cross_validation"
    if !ispath(cv_path)
        println("mkpath($(cv_path))")
        mkpath(cv_path)
    end
    path = "$(cv_path)/per_fold_avgMAPs.csv"
    writedlm(path, cv_res.per_fold_avgMAPs)

    path = "$(cv_path)/per_fold_MAPs.csv"
    writedlm(path, cv_res.per_fold_MAPs)

    path = "$(cv_path)/in_sample_LL_list.csv"
    writedlm(path, cv_res.in_sample_LL_list)

    path = "$(cv_path)/out_of_sample_LL_list.csv"
    writedlm(path, cv_res.out_of_sample_LL_list)

    path = "$(cv_path)/out_of_sample_LL_SUM.csv"
    writedlm(path, sum(cv_res.out_of_sample_LL_list[:,1]))

    # keep track of the subject ID's
    for i_fold = 1:length(f_spec)
        path = "$(cv_path)/subj_id_In_$(i_fold).csv"
        writedlm(path, f_spec[i_fold].in_sample_ids)
        path = "$(cv_path)/subj_id_Out_$(i_fold).csv"
        writedlm(path, f_spec[i_fold].out_of_sample_ids)
    end

    path = "$(cv_path)/LL_out_subjectwise.csv"
    writedlm(path, cv_res.LL_out_subjectwise)

end
