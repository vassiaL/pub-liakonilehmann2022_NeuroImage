using PyPlot
using Statistics
using LinearAlgebra

""" Function for results of Fig. A.1 """
function RPE_SPE_corr(;results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits/"),
                        algo_rpe = "sarsa_lambda",
                        algo_spe = "forward_learner",
                        fitType = "pop",
                        subject_of_interest = 3,
                        )

    save_path = makehomesavepath("RPE_SPE_corr")
    mkdir(save_path)

    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"

    dataFmriSubj = 0
    rpe_pop = 0
    surprise_trial_idx = 0
    all_RPE_surprise = Array{Float64,1}()
    all_SPE_surprise = Array{Float64,1}()
    all_RPE = Array{Float64,1}()
    all_SPE = Array{Float64,1}()
    all_RPE_nonsurprise = Array{Float64,1}()
    all_SPE_nonsurprise = Array{Float64,1}()
    corrs = []
    corrs_non_surprise = []
    fig = 0
    axes = 0

    n_actions_persubject = []
    n_surprise_trials_persubj = []

    n_actions_subject = 0
    n_actions_total = 0
    n_surprise_trials_subj = 0

    fig, axes = subplots(nrows=1, ncols=3, figsize=(12,3))

    for i_subj in 1:21

        df_rpe = getSubjSignal(subjID = i_subj,
                        results_fit_path = results_fit_path,
                        signalname = "RPE",
                        fitType = fitType,
                        algoname = algo_rpe)

        df_spe = getSubjSignal(subjID = i_subj,
                        results_fit_path = results_fit_path,
                        signalname = "SPE",
                        # signalname = "Surprise",
                        fitType = fitType,
                        algoname = algo_spe)

        correlation = cor(df_rpe[:,1], df_spe[:,1])
        push!(corrs, correlation)

        (dataFmriSubj, stats) = RL_Fit.load_SARSPEI(data_path; subject_id = i_subj)

        surprise_trial_idx = findall(dataFmriSubj[:,8] .== 1) .-1
        # NOTE:
        # Shifting -1 is correct.
        # In the csv files the surprise indicator is placed on the landing state
        # But to calculate the signals we take the tuple (s,a,s'), ie the column of the csv file
        # corresponding to the current state. So the RPE calculated on the surprise trial is
        # placed at -1 the index of the surprise trial.

        all_RPE = vcat(all_RPE, df_rpe[:,1])
        all_SPE = vcat(all_SPE, df_spe[:,1])

        all_RPE_surprise= vcat(all_RPE_surprise, df_rpe[:,1][surprise_trial_idx])
        all_SPE_surprise = vcat(all_SPE_surprise, df_spe[:,1][surprise_trial_idx])

        n_actions_subject = length(df_rpe[:,1])
        n_surprise_trials_subj = length(surprise_trial_idx)

        push!(n_actions_persubject, n_actions_subject)
        push!(n_surprise_trials_persubj, n_surprise_trials_subj)

        non_surprise = setdiff(1:n_actions_subject, surprise_trial_idx)

        corr_non_surprise = cor(df_rpe[:,1][non_surprise], df_spe[:,1][non_surprise])

        push!(corrs_non_surprise, corr_non_surprise)

        all_RPE_nonsurprise= vcat(all_RPE_nonsurprise, df_rpe[:,1][non_surprise])
        all_SPE_nonsurprise = vcat(all_SPE_nonsurprise, df_spe[:,1][non_surprise])

        n_actions_total += length(df_rpe[:,1])
        if i_subj == subject_of_interest
            axes[1].scatter(df_rpe[:,1], df_spe[:,1])
            axes[1].set_xlabel("RPE")
            axes[1].set_ylabel("SPE")
            axes[1].set_title("subj: $fitType, Cor(RPE, SPE) = $(round(correlation,digits=3)) [$(round(corr_non_surprise,digits=3))], #S-trials:$n_surprise_trials_subj/$n_actions_subject")
            axes[1].scatter(df_rpe[:,1][surprise_trial_idx], df_spe[:,1][surprise_trial_idx], color="r")
            axes[1].grid()
            axes[1].set_xlim((-1, +1) )
            axes[1].set_ylim((0., +1) )
        end

    end
    fig.suptitle("SPE/RPE correlations. Param: $fitType. #S-trials: $(length(all_SPE_surprise))/$n_actions_total", y=1.)

    axes[2].hist(all_RPE_surprise, alpha=0.9, color="r", edgecolor="black", density=true)#,normed=true)
    axes[3].hist(all_SPE_surprise, alpha=0.9, color="r", edgecolor="black", density=true)#,normed=true)
    axes[2].set_title("Histogram of RPE at surprise trials")
    axes[3].set_title("Histogram of SPE at surprise trials")

    axes[3].hist(all_SPE_nonsurprise, color="blue", alpha=0.5, edgecolor="black", density=true)#,normed=true)

    fig.tight_layout()

    fig.savefig(joinpath(save_path, "Hist_RPE_SPE.pdf"))   # save the figure to file

    abs_corrs = abs.(corrs)
    abs_corrs_non_surprise = abs.(corrs_non_surprise)

    println("mean abs(corr): $(round(mean(abs_corrs),digits=3))")
    println("std corr: $(round(std(abs_corrs),digits=3))")
    println("minimum corr: $(round(minimum(corrs),digits=3))")
    println("maximum corr: $(round(maximum(corrs),digits=3))")

    println("mean abs(corr_non_surprise): $(round(mean(abs_corrs_non_surprise),digits=3))")
    println("std corr_non_surprise: $(round(std(abs_corrs_non_surprise),digits=3))")
    println("minimum corr_non_surprise: $(round(minimum(corrs_non_surprise),digits=3))")
    println("maximum corr_non_surprise: $(round(maximum(corrs_non_surprise),digits=3))")

    all_RPE_surprise, all_SPE_surprise, corrs, corrs_non_surprise,
    n_actions_persubject, n_surprise_trials_persubj
end


""" Function for results of Fig. A.2 """
function signals_corr(;
                        results_fit_path =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits/"),
                         algoname1 = "hybrid_actorcritic_FWD",
                          algoname2 = "hybrid_actorcritic_FWD",
                        signalname1 = "RPE",
                        signalname2 = "SPE",
                        fitType = "pop",
                        subject_of_interest = 3)

    save_path = makehomesavepath("signals_corr")
    mkdir(save_path)

    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv"

    dataFmriSubj = 0
    rpe_pop = 0
    surprise_trial_idx = 0
    all_signal1_surprise = Array{Float64,1}()
    all_signal2_surprise = Array{Float64,1}()
    all_signal1 = Array{Float64,1}()
    all_signal2 = Array{Float64,1}()
    all_signal1_nonsurprise = Array{Float64,1}()
    all_signal2_nonsurprise = Array{Float64,1}()

    all_signal1_surprise_DM = Array{Float64,1}()
    all_signal2_surprise_DM = Array{Float64,1}()
    all_signal1_nonsurprise_DM = Array{Float64,1}()
    all_signal2_nonsurprise_DM = Array{Float64,1}()

    corrs = []
    corrs_non_surprise = []
    fig = 0
    axes = 0

    n_actions_persubject = []
    n_surprise_trials_persubj = []

    n_actions_subject = 0
    n_actions_total = 0
    n_surprise_trials_subj = 0

    fig, axes = subplots(nrows=1, ncols=3, figsize=(9,3))

    for i_subj in 1:21

        df_signal1 = getSubjSignal(subjID = i_subj,
                        results_fit_path = results_fit_path,
                        signalname = signalname1,
                        fitType = fitType,
                        algoname = algoname1)

        df_signal2 = getSubjSignal(subjID = i_subj,
                        results_fit_path = results_fit_path,
                        signalname = signalname2,
                        # signalname = "Surprise",
                        fitType = fitType,
                        algoname = algoname2)

        # --- Signals the way they are
        signal1 = df_signal1[:,1]
        signal2 = df_signal2[:,1]

        # --- Mean center everyone
        signal1_DM = meancenter(df_signal1[:,1])
        signal2_DM = meancenter(df_signal2[:,1])

        correlation = cor(signal1_DM, signal2_DM)
        @show correlation
        push!(corrs, correlation)
        all_signal1 = vcat(all_signal1, signal1)
        all_signal2 = vcat(all_signal2, signal2)

        (dataFmriSubj, stats) = RL_Fit.load_SARSPEI(data_path; subject_id = i_subj)

        surprise_trial_idx = findall(dataFmriSubj[:,8] .== 1) .-1
        # NOTE:
        # Shifting -1 is correct.
        # In the csv files the surprise indicator is placed on the landing state
        # But to calculate the signals we take the tuple (s,a,s'), ie the column of the csv file
        # corresponding to the current state. So the RPE calculated on the surprise trial is
        # placed at -1 the index of the surprise trial.

        # --- Signals the way they are
        signal1surprise = signal1[surprise_trial_idx]
        signal2surprise = signal2[surprise_trial_idx]
        # --- Demeaned
        signal1surprise_DM = signal1_DM[surprise_trial_idx]
        signal2surprise_DM = signal2_DM[surprise_trial_idx]

        all_signal1_surprise= vcat(all_signal1_surprise, signal1surprise)
        all_signal2_surprise = vcat(all_signal2_surprise, signal2surprise)

        all_signal1_surprise_DM= vcat(all_signal1_surprise_DM, signal1surprise_DM)
        all_signal2_surprise_DM = vcat(all_signal2_surprise_DM, signal2surprise_DM)

        n_actions_subject = length(signal1)
        n_surprise_trials_subj = length(surprise_trial_idx)

        push!(n_actions_persubject, n_actions_subject)
        push!(n_surprise_trials_persubj, n_surprise_trials_subj)
        non_surprise = setdiff(1:n_actions_subject, surprise_trial_idx)

        # --- Signals the way they are
        signal1nonsurprise = signal1[non_surprise]
        signal2nonsurprise = signal2[non_surprise]
        # --- Demeaned
        signal1nonsurprise_DM = signal1_DM[non_surprise]
        signal2nonsurprise_DM = signal2_DM[non_surprise]

        correlation_non_surprise = cor(signal1nonsurprise_DM, signal2nonsurprise_DM)
        push!(corrs_non_surprise, correlation_non_surprise)

        all_signal1_nonsurprise= vcat(all_signal1_nonsurprise, signal1nonsurprise)
        all_signal2_nonsurprise = vcat(all_signal2_nonsurprise, signal2nonsurprise)

        all_signal1_nonsurprise_DM= vcat(all_signal1_nonsurprise_DM, signal1nonsurprise_DM)
        all_signal2_nonsurprise_DM = vcat(all_signal2_nonsurprise_DM, signal2nonsurprise_DM)

        n_actions_total += length(signal1)
        if i_subj == subject_of_interest
            axes[1].scatter(signal1, signal2)
            axes[1].set_xlabel(signalname1)
            axes[1].set_ylabel(signalname2)
            axes[1].set_title("subj: $fitType, Cor($signalname1, $signalname2) = $(round(correlation,digits=3)) [$(round(correlation_non_surprise,digits=3))], #S-trials:$n_surprise_trials_subj/$n_actions_subject")
            axes[1].scatter(signal1surprise, signal2surprise, color="r")
            axes[1].grid()
            axes[1].set_xlim((-1, +1) )
            axes[1].set_ylim((0., +1) )
        end

    end
    fig.suptitle("$signalname2/$signalname1 correlations. Param: $fitType. #S-trials: $(length(all_signal2_surprise))/$n_actions_total", y=1.)

    nbins = 15

    axes[2].hist(all_signal1_surprise, alpha=0.9, color="r", edgecolor="black", density=true) #, normed=true)
    axes[3].hist(all_signal2_surprise, alpha=0.9, color="r", edgecolor="black", density=true, bins=nbins)#, normed=true)

    axes[2].set_title("Histogram of $signalname1 at surprise trials")
    axes[3].set_title("Histogram of $signalname2 at surprise trials")

    axes[3].hist(all_signal2_nonsurprise, color="blue", alpha=0.5, edgecolor="black", density=true, bins=nbins)#, normed=true)

    fig.tight_layout()
    fig.savefig(joinpath(save_path, "Hist_$signalname1$signalname2.pdf"))   # save the figure to file

    abs_corrs = abs.(corrs)
    abs_corrs_non_surprise = abs.(corrs_non_surprise)

    println("mean corr: $(round(mean(corrs),digits=3))")
    println("std  corr: $(round(std(corrs),digits=3))")
    println("mean abs(corr): $(round(mean(abs_corrs),digits=3))")
    println("std  abs(corr): $(round(std(abs_corrs),digits=3))")
    println("minimum corr: $(round(minimum(corrs),digits=3))")
    println("maximum corr: $(round(maximum(corrs),digits=3))")

    println("mean corr_non_surprise: $(round(mean(corrs_non_surprise),digits=3))")
    println("std  corr_non_surprise: $(round(std(corrs_non_surprise),digits=3))")
    println("mean abs(corr_non_surprise): $(round(mean(abs_corrs_non_surprise),digits=3))")
    println("std  abs(corr_non_surprise): $(round(std(abs_corrs_non_surprise),digits=3))")
    println("minimum corr_non_surprise: $(round(minimum(corrs_non_surprise),digits=3))")
    println("maximum corr_non_surprise: $(round(maximum(corrs_non_surprise),digits=3))")

    all_signal1_surprise, all_signal2_surprise, corrs, corrs_non_surprise,
    n_actions_persubject, n_surprise_trials_persubj
end

function cosinesimilarity(a,b)
    sum(a.*b) / (sqrt(sum(a.^2)*sum(b.^2)))
end

function meancenter(a)
    a = a .- mean(a)
end
