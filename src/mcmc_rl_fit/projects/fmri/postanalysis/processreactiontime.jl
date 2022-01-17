using DataFrames, DataFramesMeta, CSV
using PGFPlots
using Statistics, HypothesisTests

""" Functions for results of Fig.1B and Fig.1C """
# Get mean RT per subject and averaged across subjects
# for surprise and non-surprise trials
# Check statistical significance of difference
function processreactiontime(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/reactiontime_persubj/"),
        )

    savepath = makehomesavepath("processreactiontime")
    mkdir(savepath)

    subjectfiles = readdir(filepath)

    dfresults = DataFrame(subjectname = String[],
                    mean_rt_surprise = Float64[],
                    std_rt_surprise = Float64[],
                    se_rt_surprise = Float64[],
                    mean_rt_nonsurprise = Float64[],
                    std_rt_nonsurprise = Float64[],
                    se_rt_nonsurprise = Float64[],
                    mean_rt_diff = Float64[],
                    )

    dfresults_after4 = DataFrame(subjectname = String[],
                    mean_rt_surprise = Float64[],
                    std_rt_surprise = Float64[],
                    se_rt_surprise = Float64[],
                    mean_rt_nonsurprise = Float64[],
                    std_rt_nonsurprise = Float64[],
                    se_rt_nonsurprise = Float64[],
                    mean_rt_diff = Float64[],
                    )
    dfresult_scatter = DataFrame()
    dfresult_scatter_after4 = DataFrame()
    dfresult_scatter_diff = DataFrame()
    dfresult_scatter_diff_after4 = DataFrame()
    for i in subjectfiles
        @show i
        # subjectname = split(split(i, "_")[end], ".")[end-1]
        dfalgo = CSV.read(joinpath(filepath, i), header=false)
        rt_surp = @where(dfalgo, :Column2 .== 1)[!, :Column1]
        rt_nonsurp = @where(dfalgo, :Column2 .== 0)[!, :Column1]

        # Remove the first one cause some subjects had their eyes closed
        # and did not realize the experiment has started:
        rt_nonsurp = rt_nonsurp[2:end]

        rt_surp = rt_removeoutliers(rt_surp, showstr = "surp")
        rt_nonsurp = rt_removeoutliers(rt_nonsurp, showstr = "non_surp")
        # ---------
        #
        mean_rt_surprise, std_rt_surprise, se_rt_surprise = calc_stats_nan(reshape(rt_surp, :, 1)') # reshape to make it 2-d (datapoints x 1) and transpose
        mean_rt_nonsurprise, std_rt_nonsurprise, se_rt_nonsurprise = calc_stats_nan(reshape(rt_nonsurp, :, 1)') # reshape to make it 2-d (datapoints x 1) and transpose

        push!(dfresults, [i, mean_rt_surprise[1], std_rt_surprise[1], se_rt_surprise[1],
                                        mean_rt_nonsurprise[1], std_rt_nonsurprise[1], se_rt_nonsurprise[1],
                                        mean_rt_surprise[1] - mean_rt_nonsurprise[1]
                                        ])
        # In columns: first nonsurprise, then surprise
        dfresult_scatter[!, Symbol(i*"_mean_rt")] = [mean_rt_nonsurprise[1], mean_rt_surprise[1]]
        dfresult_scatter[!, Symbol(i*"_std_rt")] = [std_rt_nonsurprise[1], std_rt_surprise[1]]
        dfresult_scatter[!, Symbol(i*"_se_rt")] = [se_rt_nonsurprise[1], se_rt_surprise[1]]

        # ----------------------------------------------------------------------
        # Remove the data points before the 5th episode (introduction of surprise trials)
        # ----------------------------------------------------------------------
        rt_surp_after4 = @where(dfalgo, :Column2 .== 1, :Column3 .> 4)[!, :Column1]
        rt_nonsurp_after4 = @where(dfalgo, :Column2 .== 0, :Column3 .> 4)[!, :Column1]

        # ----- Remove outliers ----
        rt_surp_after4 = rt_removeoutliers(rt_surp_after4, showstr = "surp - after4")
        rt_nonsurp_after4 = rt_removeoutliers(rt_nonsurp_after4, showstr = "non_surp - after4")
        # ---------


        mean_rt_surprise_after4, std_rt_surprise_after4, se_rt_surprise_after4 = calc_stats_nan(reshape(rt_surp_after4, :, 1)') # reshape to make it 2-d (datapoints x 1) and transpose
        mean_rt_nonsurprise_after4, std_rt_nonsurprise_after4, se_rt_nonsurprise_after4 = calc_stats_nan(reshape(rt_nonsurp_after4, :, 1)') # reshape to make it 2-d (datapoints x 1) and transpose

        push!(dfresults_after4, [i, mean_rt_surprise_after4[1], std_rt_surprise_after4[1], se_rt_surprise_after4[1],
                                        mean_rt_nonsurprise_after4[1], std_rt_nonsurprise_after4[1], se_rt_nonsurprise_after4[1],
                                        mean_rt_surprise_after4[1] - mean_rt_nonsurprise_after4[1]
                                        ])
        # In columns: first nonsurprise, then surprise
        dfresult_scatter_after4[!, Symbol(i*"_mean_rt")] = [mean_rt_nonsurprise_after4[1], mean_rt_surprise_after4[1]]
        dfresult_scatter_after4[!, Symbol(i*"_std_rt")] = [std_rt_nonsurprise_after4[1], std_rt_surprise_after4[1]]
        dfresult_scatter_after4[!, Symbol(i*"_se_rt")] = [se_rt_nonsurprise_after4[1], se_rt_surprise_after4[1]]

    end
    CSV.write(joinpath(savepath, "reactiontime_stats.csv"), dfresults, delim = " ");
    CSV.write(joinpath(savepath, "reactiontime_stats_scatter.csv"), dfresult_scatter, delim = " ");

    CSV.write(joinpath(savepath, "reactiontime_stats_after4.csv"), dfresults_after4, delim = " ");
    CSV.write(joinpath(savepath, "reactiontime_stats_scatter_after4.csv"), dfresult_scatter_after4, delim = " ");

    println("All non-surprise trials:")
    @show OneSampleTTest(dfresults[!,:mean_rt_surprise], dfresults[!,:mean_rt_nonsurprise])
    pval = pvalue(OneSampleTTest(dfresults[!,:mean_rt_surprise], dfresults[!,:mean_rt_nonsurprise]))
    println(" ------ ")
    println("After removing trials before the 5th episode:")
    @show OneSampleTTest(dfresults_after4[!,:mean_rt_surprise], dfresults_after4[!,:mean_rt_nonsurprise])
    pval_after4 = pvalue(OneSampleTTest(dfresults_after4[!,:mean_rt_surprise], dfresults_after4[!,:mean_rt_nonsurprise]))

    @show length(dfresults[:,:mean_rt_surprise])
    @show sqrt(length(dfresults[:,:mean_rt_surprise]))
    dfresults_meanall = DataFrame()
    dfresults_meanall[!, Symbol("mean_rt_surprise")] = [mean(dfresults[:,:mean_rt_surprise])]
    dfresults_meanall[!, Symbol("std_rt_surprise")] = [std(dfresults[:,:mean_rt_surprise])]
    dfresults_meanall[!, Symbol("se_rt_surprise")] = [std(dfresults[:,:mean_rt_surprise]) / sqrt(length(dfresults[:,:mean_rt_surprise]))]
    dfresults_meanall[!, Symbol("mean_rt_nonsurprise")] = [mean(dfresults[:,:mean_rt_nonsurprise])]
    dfresults_meanall[!, Symbol("std_rt_nonsurprise")] = [std(dfresults[:,:mean_rt_nonsurprise])]
    dfresults_meanall[!, Symbol("se_rt_nonsurprise")] = [std(dfresults[:,:mean_rt_nonsurprise]) / sqrt(length(dfresults[:,:mean_rt_nonsurprise]))]
    #
    dfresults_meanall_after4 = DataFrame()
    dfresults_meanall_after4[!, Symbol("mean_rt_surprise")] = [mean(dfresults_after4[:,:mean_rt_surprise])]
    dfresults_meanall_after4[!, Symbol("std_rt_surprise")] = [std(dfresults_after4[:,:mean_rt_surprise])]
    dfresults_meanall_after4[!, Symbol("se_rt_surprise")] = [std(dfresults_after4[:,:mean_rt_surprise]) / sqrt(length(dfresults_after4[:,:mean_rt_surprise]))]
    dfresults_meanall_after4[!, Symbol("mean_rt_nonsurprise")] = [mean(dfresults_after4[:,:mean_rt_nonsurprise])]
    dfresults_meanall_after4[!, Symbol("std_rt_nonsurprise")] = [std(dfresults_after4[:,:mean_rt_nonsurprise])]
    dfresults_meanall_after4[!, Symbol("se_rt_nonsurprise")] = [std(dfresults_after4[:,:mean_rt_nonsurprise]) / sqrt(length(dfresults_after4[:,:mean_rt_nonsurprise]))]

    CSV.write(joinpath(savepath, "reactiontime_meanall.csv"), dfresults_meanall, delim = " ");

    CSV.write(joinpath(savepath, "reactiontime_meanall_after4.csv"), dfresults_meanall_after4, delim = " ");

    dfresults, pval, dfresults_after4, pval_after4,
    dfresult_scatter, dfresult_scatter_after4,
    dfresults_meanall, dfresults_meanall_after4
end


function rt_removeoutliers(reactiontimes;
                            thrs = 8.,
                            showstr = "surp"
                            )
    k  = findall(reactiontimes .> thrs)
    # @show rt_surp[ksurp]
    if ~isempty(k)
        @show k
        println(showstr)
        @show reactiontimes[k]
    end
    reactiontimes = deleteat!(reactiontimes, k)

    reactiontimes
end
