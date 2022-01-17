using CSV
using DataFrames
using PGFPlots
using Statistics

"""
Plotting signals of 2 algorithms for visual comparison (similar to Fig. A.5)

NOTE:
-----
A lot of manual setting of things... signalnames, algonames, legendentry,
scalesurprise, ymax, xmax, etc...
"""

function compare_MB_signals(;
            results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits"),
            signalname1 = "SPE",
            # signalname1 = "gamma_surprise",
            # signalname1 = "Surprise",
            # signalname1 = "RPE",
            # signalname2 = "RPE",
            signalname2 = "Surprise",
            # signalname2 = "DeltaPolicy",
            # signalname2 = "SPE",
            # algoname1 = "forward_learner",
            algoname1 = "hybrid_actorcritic_FWD",
            # algoname1 = "actor_critic_TD_modulatedActor_1degree",
            # algoname1 = "sarsa_lambda",
            algoname2 = "actor_critic_TD_modulatedActor_1degree",
            fitType = "pop"
            )
    save_path1 = makehomesavepath("M_signalsInTime")
    mkdir(save_path1)
    save_path2 = makehomesavepath("M_signalsVsEachOther")
    mkdir(save_path2)
    save_path3 = makehomesavepath("M_signalsDifferenceInTime")
    mkdir(save_path3)

    for i in 1:21
        df_signal1, df_signal2 = getSubjSignalofTwoAlgos(subjID = i,
                results_fit_path = results_fit_path,
                signalname1 = signalname1,
                signalname2 = signalname2,
                algoname1 = algoname1,
                algoname2 = algoname2,
                fitType = fitType)
        df_surprise = getSubjSurpriseEvents(subjID = i)

        plotSubjSignalsInTime(df_signal1, df_signal2, df_surprise,
                         subjID = i,
                         scalesurprise = 3, #1.5, #3, #1.3, #1,#3, #1, #10, #1,
                         signalname1 = signalname1,
                         signalname2 = signalname2,
                         save_path = save_path1)
        plotSubjSignalsVsEachOther(df_signal1, df_signal2,
                        subjID = i,
                        signalname1 = signalname1,
                        signalname2 = signalname2,
                        save_path = save_path2)
        # plotSubjSignalDifferenceInTime(df_signal1, df_signal2, df_surprise,
        #                 subjID = i,
        #                 signalname1 = signalname1,
        #                 signalname2 = signalname2,
        #                 scalesurprise = 0.7,
        #                 save_path = save_path3)
    end
end


function getSubjSignalofTwoAlgos(;subjID = 3,
        results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits"),
        signalname1 = "SPE",
        signalname2 = "Surprise",
        algoname1 = "hybrid_actorcritic_FWD",
        algoname2 = "actor_critic_TD_modulatedActor_1degree",
        fitType = "pop",
        )
    df_signal1 = getSubjSignal(subjID = subjID,
                    results_fit_path = results_fit_path,
                    signalname = signalname1,
                    fitType = fitType,
                    algoname = algoname1)
    df_signal2 = getSubjSignal(subjID = subjID,
                    results_fit_path = results_fit_path,
                    signalname = signalname2,
                    fitType = fitType,
                    algoname = algoname2)

    df_signal1, df_signal2
end

function getSubjSignal(;subjID = 3,
                results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits"),
                signalname = "SPE",
                algoname = "actor_critic_TD_modulatedActor_1degree",
                fitType = "pop",
                )
    df_signal = CSV.read(joinpath(results_fit_path, algoname, "maxLL_fit",
                    "subj_"*string(subjID), signalname*"_"*fitType*".csv"),
                    header=false)
    rename!(df_signal, Dict(:Column1 => Symbol(algoname)))
end
function getSubjSurpriseEvents(;subjID = 3,
                results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/all_events"),
                eventname = "is_surprise_trial")
    df_event = getSubjEvent(;subjID = subjID,
                    results_fit_path = results_fit_path,
                    eventname = eventname)
    rename!(df_event, Dict(:Column1 => Symbol("is_surprise_trial")))
end
function getSubjEvent(;subjID = 3,
                results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/all_events"),
                eventname = "is_surprise_trial")
    df_event = CSV.read(joinpath(results_fit_path,
                    "subj_"*string(subjID), eventname*".csv"),
                    header=false)
end

function plotSubjSignalsInTime(df_signal1, df_signal2;
                    subjID = 3,
                    signalname1 = "SPE",
                    signalname2 = "Surprise",
                    save_path = "./",
                    )
        legendentry1 = replace(signalname1, "_"=>"") * ": " * replace(string(names(df_signal1)[1]), "_"=>"") # Remove "_" cause latex problem
        legendentry2 = replace(signalname2, "_"=>"") * ": " * replace(string(names(df_signal2)[1]), "_"=>"") # Remove "_" cause latex problem

        ax = Axis( #1:size(df_signal1,1)
            [Plots.Scatter(1:size(df_signal1,1), df_signal1[:,1],
                    legendentry = legendentry1,
                    markSize=1),
                    # , style = "violet!95!black"),
            Plots.Scatter(1:size(df_signal2,1), df_signal2[:,1],
                    legendentry = legendentry2,
                    markSize=1)],
           title = replace(signalname1, "_"=>"") *" and "*replace(signalname2, "_"=>""),
           # legendStyle = "{at={(1.05,1.0)},anchor=north west}")
           legendStyle = "{at={(0.3,-0.1)},anchor=north west}",style = "grid=both")
           # ymin = -0.1, ymax = 0.1)
        save(joinpath(save_path, "subj_"*string(subjID)*"_"*signalname1*"_"*signalname2*".pdf"), ax)
end
function plotSubjSignalsInTime(df_signal1, df_signal2, df_surprise;
                    subjID = 3,
                    scalesurprise = 1, #10
                    signalname1 = "SPE",
                    signalname2 = "Surprise",
                    save_path = "./",
                    )

        legendentry1 = L"SPE$_{\mathrm{FW}}$"

        legendentry2 = L"$\textbf{S}_{\mathrm{BF}}$"

        signal1 = df_signal1[:,1]
        signal2 = df_signal2[:,1] #.* 100

        ax = Axis( #1:size(df_signal1,1)
            [Plots.Scatter(1:size(df_signal1,1), signal1,
                    # mark = *,
                    legendentry = legendentry1,
                    markSize=1,
                    style = "orange, mark options={fill=orange}")
                    # scatterClasses=sc1)
            Plots.Scatter(1:size(df_signal2,1), signal2, #df_signal2[:,1],
                    legendentry = legendentry2,
                    # legendentry = legendentry2 * "Norm",
                    markSize=1,
                    style = "violet!95!black, mark options={fill=violet!95!black}")
                    # This is correct, cause it shifts surprise trials one step earlier (which correct given the way things are calculated
                    # and are encoded in all_data, all_stats = RL_Fit.load_SARSPEI(data_path))
            Plots.Linear(1:size(df_surprise,1)-1, scalesurprise .* df_surprise[2:end,1], style=:ycomb, mark = :none)
                    ],
           legendStyle = "{at={(0.01,0.95)},anchor=north west}",
           # ymin = -0.1, ymax = 1.,
           ymax = 3.1,
           # ymax = 1.6,
           xlabel= "time steps",
           style="ymajorgrids=true")
        save(joinpath(save_path, "subj_"*string(subjID)*"_"*signalname1*"_"*signalname2*".pdf"), ax)
end


function plotSubjSignalsVsEachOther(df_signal1, df_signal2;
                    subjID = 3,
                    signalname1 = "SPE",
                    signalname2 = "Surprise",
                    save_path = "./",
                    )

        legendentry1 = L"SPE$_{\mathrm{FW}}$ - Hybrid Actor-critic"
        legendentry2 = L"$\textbf{S}_{\mathrm{BF}}$ - Surprise Actor-critic"

        signal1 = df_signal1[:,1]
        signal2 = df_signal2[:,1]

        # # --- Theory curve
        x = collect(0.:0.05:0.95)
        y = 1. ./ (7. .* (1. .- x) )

        ax = Axis(
            [Plots.Scatter(signal1, signal2,
                    # legendentry = legendentry1,
                    markSize=1,
                    style = "black, mark options={fill=black}"
                    )
                    # ],
            ##### Plot theory curve of Sbf vs SPE
            Plots.Linear(x, y,
                    # legendentry = legendentry1,
                    # markSize=1,
                    style = "red, thick, mark = :none"
                    )
                    ],
           xlabel=legendentry1,
           ylabel=legendentry2,
           xmin = 0.,
           ymin = 0.,
           ymax=3., # for Surprise
           xmax=1.03,
           style = "grid=both"
           )
           # ymin = -0.1, ymax = 0.1)
        save(joinpath(save_path, "subj_"*string(subjID)*"_"*signalname1*"_"*signalname2*".pdf"), ax)
end


function plotSubjSignalDifferenceInTime(df_signal1, df_signal2;
                    subjID = 3,
                    signalname1 = "SPE",
                    signalname2 = "Surprise",
                    save_path = "./",
                    )
        legendentry1 = replace(string(names(df_signal1)[1]), "_"=>"") # Remove "_" cause latex problem
        legendentry2 = replace(string(names(df_signal2)[1]), "_"=>"") # Remove "_" cause latex problem

        ax = Axis(
            [Plots.Scatter(1:length(df_signal1[:,1]) , df_signal1[:,1] .- df_signal2[:,1],
                    markSize=1)],
           title = "Difference " * replace(signalname1, "_"=>"") *" - "*replace(signalname2, "_"=>""),
           xlabel="Time",
           ylabel="Difference",
           style = "grid=both"
           )

        save(joinpath(save_path, "subj_"*string(subjID)*"_"*signalname1*"_"*signalname2*".pdf"), ax)
end
function plotSubjSignalDifferenceInTime(df_signal1, df_signal2, df_surprise;
                    subjID = 3,
                    signalname1 = "RPE",
                    signalname2 = "RPE",
                    scalesurprise = 1.,
                    save_path = "/Users/vasia/Desktop/",
                    )
        legendentry1 = replace(signalname1, "_"=>"") * ": " * replace(string(names(df_signal1)[1]), "_"=>"") # Remove "_" cause latex problem
        legendentry2 = replace(signalname2, "_"=>"") * ": " * replace(string(names(df_signal2)[1]), "_"=>"") # Remove "_" cause latex problem

        ax = Axis(
            [Plots.Scatter(1:length(df_signal1[:,1]) , df_signal1[:,1] .- df_signal2[:,1],
                    # legendentry = legendentry1,
                    markSize=1),
            # This is correct, cause it shifts surprise trials one step earlier (which correct given the way things are calculated
            # and are encoded in all_data, all_stats = RL_Fit.load_SARSPEI(data_path))
            Plots.Linear(1:size(df_surprise,1)-1, scalesurprise .* df_surprise[2:end,1], style=:ycomb, mark = :none)
            ],
           title = "Difference " * replace(signalname1, "_"=>"") *" - "*replace(signalname2, "_"=>""),
           xlabel="Time",
           ylabel="Difference",
           style = "grid=both"
           )

        save(joinpath(save_path, "subj_"*string(subjID)*"_"*signalname1*"_"*signalname2*".pdf"), ax)
end
