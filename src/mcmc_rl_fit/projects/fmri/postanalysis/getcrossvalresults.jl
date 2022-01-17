using DataFrames, DataFramesMeta, CSV
using PGFPlots
using Statistics

""" Functions for Fig.3 """

# ---- Average crossvalidated LL across multiple runs
# For results of Fig. 3
function getCrossValResults_multipleruns_incolumnsandrows(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/crossvalidation/"),
        filestr = "",
        # savepath = "./"
        )
    savepath = makehomesavepath("getCrossValResults")
    mkdir(savepath)

    prefix = split(filepath, "/")[end-1]
    runs = getsubfolders(filepath)
    dfresults = DataFrame()
    dfresults[!, :runs] = 1:length(runs)
    for irun in 1:length(runs)
        algorithms = getsubfolders(joinpath(filepath, runs[irun]))
        for i in algorithms
            if irun == 1 # Initialize
                 dfresults[!, Symbol(i)] .= 0.0
            end
            dfalgo = CSV.read(joinpath(filepath, runs[irun], i, "cross_validation/out_of_sample_LL_SUM.csv"), header=false)
            dfresults[irun, Symbol(i)] = dfalgo[1, :Column1]
        end
    end
    CSV.write(joinpath(savepath, prefix*"_resultscrossval_multipleruns"*filestr*".csv"), dfresults,
            delim = " ");

    dfresultsmean = DataFrame()
    algorithms = getsubfolders(joinpath(filepath, runs[1]))
    for i in algorithms
        dfresultsmean[!, Symbol(i*"_mean")] = [mean(dfresults[!, Symbol(i)])]
        dfresultsmean[!, Symbol(i*"_std")] = [std(dfresults[!, Symbol(i)])]
        dfresultsmean[!, Symbol(i*"_stderror")] = [std(dfresults[!, Symbol(i)]) / sqrt(length(dfresults[!, Symbol(i)]))]
    end
    CSV.write(joinpath(savepath, prefix*"_resultscrossval_multipleruns_mean"*filestr*".csv"), dfresultsmean,
            delim = " ");

    dfrows = describe(dfresults, :mean, :std)
    CSV.write(joinpath(savepath, prefix*"_resultscrossval_multipleruns_mean"*filestr*"_rows.csv"), dfrows,
                   delim = " ");

    dfresults, dfresultsmean
end

# Transform the Klaas results that are saved from Matlab in the form we need for
# plotting them as barplots
function getKlassresults_selection(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/KlaasAnalysis/"),
        )

    savepath = makehomesavepath("getKlassresults")
    mkdir(savepath)

    dfalgolist = CSV.read(joinpath(filepath, "algolist.csv"), header=false)

    df_selection_PG = CSV.read(joinpath(filepath, "selection_PG.csv"), header=false)
    df_pxp_PG = CSV.read(joinpath(filepath, "pxp_PG.csv"), header=false)
    df_exp_r_PG = CSV.read(joinpath(filepath, "exp_r_PG.csv"), header=false)
    df_alpha_PG = CSV.read(joinpath(filepath, "alpha_PG.csv"), header=false)

    dfresults_pxp = DataFrame()
    dfresults_exp_r = DataFrame()
    dfresults_alpha = DataFrame()
    nAlgo = size(df_selection_PG,2)
    for i in 1:nAlgo
        dfresults_pxp[!, Symbol(dfalgolist[df_selection_PG[1,i],1])] = [df_pxp_PG[1,i]]
        dfresults_exp_r[!, Symbol(dfalgolist[df_selection_PG[1,i],1])] = [df_exp_r_PG[1,i]]
        dfresults_alpha[!, Symbol(dfalgolist[df_selection_PG[1,i],1])] = [df_alpha_PG[1,i]]
    end

    CSV.write(joinpath(savepath, "resultsKlaas_pxp_PG.csv"), dfresults_pxp, delim = " ");
    CSV.write(joinpath(savepath, "resultsKlaas_exp_r_PG.csv"), dfresults_exp_r, delim = " ");
    CSV.write(joinpath(savepath, "resultsKlaas_alpha_PG.csv"), dfresults_alpha, delim = " ");
end



# ---- Average LLpop across multiple runs and calculate BICpop
# For results of Fig. A.4
function getLLpop_multiplefits_incolumnsandrows(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits_multipleruns/"),
        filestr = "",
        # savepath = "./"
        )

    savepath = makehomesavepath("getLLruns")
    mkdir(savepath)

    nr_actions = 3666
    nr_subjects = 21

    prefix = split(filepath, "/")[end-1]
    runs = getsubfolders(filepath)
    dfresultsLL = DataFrame()
    dfresultsLL[!, :runs] = 1:length(runs)
    dfresultsBICactions = DataFrame()
    dfresultsBICactions[!, :runs] = 1:length(runs)
    dfresultsBICsubj = DataFrame()
    dfresultsBICsubj[!, :runs] = 1:length(runs)
    for irun in 1:length(runs)
        algorithms = getsubfolders(joinpath(filepath, runs[irun]))
        for i in algorithms
            if irun == 1 # Initialize
                 dfresultsLL[!, Symbol(i)] .= 0.0
                 dfresultsBICactions[!, Symbol(i)] .= 0.0
                 dfresultsBICsubj[!, Symbol(i)] .= 0.0
            end
            dfalgo = CSV.read(joinpath(filepath, runs[irun], i, "maxLL_fit/LogLikelihood_MAP.csv"), header=false)
            dfalgoParams = CSV.read(joinpath(filepath, runs[irun], i, "maxLL_fit/MAP_params.csv"), header=false)
            nr_params = size(dfalgoParams,1)

            dfresultsLL[irun, Symbol(i)] = dfalgo[1, :Column1]
            dfresultsBICactions[irun, Symbol(i)] = RL_Fit.get_BIClogevidence(dfalgo[1, :Column1], nr_params, nr_actions)
            dfresultsBICsubj[irun, Symbol(i)] = RL_Fit.get_BIClogevidence(dfalgo[1, :Column1], nr_params, nr_subjects)
        end
    end
    CSV.write(joinpath(savepath, prefix*"_fitLL_multipleruns"*filestr*".csv"), dfresultsLL, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICactions_multipleruns"*filestr*".csv"), dfresultsBICactions, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICsubj_multipleruns"*filestr*".csv"), dfresultsBICsubj, delim = " ");

    dfresultsmeanLL = DataFrame()
    dfresultsmeanBICactions = DataFrame()
    dfresultsmeanBICsubjects = DataFrame()
    algorithms = getsubfolders(joinpath(filepath, runs[1]))
    for i in algorithms
        dfresultsmeanLL[!, Symbol(i*"_mean")] = [mean(dfresultsLL[!, Symbol(i)])]
        dfresultsmeanLL[!, Symbol(i*"_std")] = [std(dfresultsLL[!, Symbol(i)])]
        dfresultsmeanLL[!, Symbol(i*"_stderror")] = [std(dfresultsLL[!, Symbol(i)]) / sqrt(length(dfresultsLL[!, Symbol(i)]))]
        #
        dfresultsmeanBICactions[!, Symbol(i*"_mean")] = [mean(dfresultsBICactions[!, Symbol(i)])]
        dfresultsmeanBICactions[!, Symbol(i*"_std")] = [std(dfresultsBICactions[!, Symbol(i)])]
        dfresultsmeanBICactions[!, Symbol(i*"_stderror")] = [std(dfresultsBICactions[!, Symbol(i)]) / sqrt(length(dfresultsBICactions[!, Symbol(i)]))]
        #
        dfresultsmeanBICsubjects[!, Symbol(i*"_mean")] = [mean(dfresultsBICsubj[!, Symbol(i)])]
        dfresultsmeanBICsubjects[!, Symbol(i*"_std")] = [std(dfresultsBICsubj[!, Symbol(i)])]
        dfresultsmeanBICsubjects[!, Symbol(i*"_stderror")] = [std(dfresultsBICsubj[!, Symbol(i)]) / sqrt(length(dfresultsBICsubj[!, Symbol(i)]))]
    end

    CSV.write(joinpath(savepath, prefix*"_fitLL_multipleruns_mean"*filestr*".csv"), dfresultsmeanLL, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICactions_multipleruns_mean"*filestr*".csv"), dfresultsmeanBICactions, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICsubjects_multipleruns_mean"*filestr*".csv"), dfresultsmeanBICsubjects, delim = " ");

    dfrowsLL = describe(dfresultsLL, :mean, :std)
    dfrowsBICactions = describe(dfresultsBICactions, :mean, :std)
    dfrowsBICsubj = describe(dfresultsBICsubj, :mean, :std)
    CSV.write(joinpath(savepath, prefix*"_fitLL_multipleruns_mean"*filestr*"_rows.csv"), dfrowsLL, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICactions_multipleruns_mean"*filestr*"_rows.csv"), dfrowsBICactions, delim = " ");
    CSV.write(joinpath(savepath, prefix*"_fitBICsubjects_multipleruns_mean"*filestr*"_rows.csv"), dfrowsBICsubj, delim = " ");

    dfresultsLL, dfresultsBICactions, dfresultsBICsubj,
    dfresultsmeanLL, dfresultsmeanBICactions, dfresultsmeanBICsubjects
end
