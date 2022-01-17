# Helper functions

function generic_LL_V(data::Array{Int,2}, lfun)

    data_stats = get_data_stats(data)
    LL = 0.
    all_signals = SortedDict(Dict{Int,Step_by_step_signals}())

    for subj_id in data_stats.all_subject_ids
        subj_data = filter_by_person(data, subj_id)

        LL_subj, signals_subj = lfun(subj_id, subj_data)

        LL += LL_subj
        all_signals[signals_subj.subject_id] = signals_subj
    end
    LL, all_signals
end

function get_MAP(samples, top_n)
    idx_perm = sortperm(samples[:,end],rev=true)

    top_n = idx_perm[1:top_n]
    top_n_samples = samples[top_n,:]
    top_MAP = top_n_samples[1,:]

    top_n_avgMAP = mean(top_n_samples, dims=1)
    top_n_stdMAP = std(top_n_samples, dims=1)
    top_n_min = minimum(top_n_samples, dims=1)
    top_n_max = maximum(top_n_samples, dims=1)
    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples
end

function get_idx_marginal_posterior(mcmc_samples, idx_param_of_interest, param_Map_values, cond_width)
    # identifies the samples in mcmc_samples which are closest around the
    # marginal posterior.
    # for each parameter i (except the parameter of interest), only the
    # samples within param_Map_values[i] +- cond_width/2. are kept.
    # Used mostly for visualization purproses
    nr_params = length(param_Map_values)
    pram_indexes = setdiff(1:nr_params, idx_param_of_interest)
    m,n = size(mcmc_samples)

    idxs = trues(m)
    for i in pram_indexes
        parval_min = param_Map_values[i] - cond_width/2.
        parval_max = param_Map_values[i] + cond_width/2.
        idx_param = (mcmc_samples[:,i] .>= parval_min) .& (mcmc_samples[:,i] .<= parval_max)
        idxs .&= idx_param
    end
    return idxs
end
