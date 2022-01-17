# Run model fitting and crossvalidation
# for real or simulated data

# Main runner for model fitting and model comparison
# NOTE: These procedures take a long time to run.
# The settings are different from the ones used in the paper (DEBUG settings).
# Comment/Uncomment the corresponding lines (around 383 and 445)
# to perform the fitting and comparison procedures as in the paper.
function runner_crossval_multipletimes_fit()

    # -------- Fit (once)
    i=""
    result_path_base, batch_fit_models_spec = model_list_main(filestr = string(i))
    runner_fmri_fit(result_path_base, batch_fit_models_spec)

    # -------- Crossvalidation (multiple runs)
    nruns = 5
    for i in 1:nruns
        result_path_base, batch_fit_models_spec = model_list_main(filestr = string(i))
        runner_fmri_3foldscrossval(result_path_base,
                                        batch_fit_models_spec)
    end
end


# Main runner for model and parameter recovery
# NOTE: You need to EDIT which data (real or simulated)
# should be used in
# runner_fmri_fit and runner_fmri_3foldscrossval!!!!
# Right now the real participants' data are used.
# See lines 367 and 428
function runner_multipletimes_fit_recovery()

    # -------- Fit for parameter recovery
    # ---- We run fit multiple times to get some error bars
    nruns = 5
    for i in 1:nruns
        result_path_base, batch_fit_models_spec = model_list_paramrecovery(filestr = string(i))
        runner_fmri_fit(result_path_base, batch_fit_models_spec)

    end

    # -------- Crossvalidation (multiple runs) for model recovery
    nruns = 5
    for i in 1:nruns
        result_path_base, batch_fit_models_spec = model_list_main(filestr = string(i))
        runner_fmri_3foldscrossval(result_path_base,
                                        batch_fit_models_spec)
    end
end



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------- Model list -----------------------------------------------------------
# ------------------------------------------------------------------------------
# List and specifications of algorithms for Fig. 3 of paper
function model_list_main(; filestr = "_")

    result_path_base = makehomesavepath("test_main"*filestr)

    stochmax = 20.

    # # Biased Random Walk
    # --------------------
    biasedRandomWalk_prior = [
        RL_Fit.paramBias(0.0, 1.)
    ]
    biasedRandomWalk_prior_std = [.15]
    biasedRandomWalk_spec = RL_Fit.Model_spec_per_subj("Biased Random Walk (0-Model)",
                                RL_Fit.getLL_BiasedRandomWalk_V,
                                biasedRandomWalk_prior, biasedRandomWalk_prior_std,
                                string(result_path_base, "biased_random_walk/"))
    # Sarsa-λ
    # -------
    sarsaL_prior = [
        RL_Fit.paramAlpha(0.0, 1.0),
        RL_Fit.paramGamma(0.0, 1.0),
        RL_Fit.paramLambda(0.0, 1.0),
        RL_Fit.paramTemp(0.01, 10.)
        ]
    sarsaL_prior_std = [.15, .15, .15, 1.5]
    sarsaL_spec = RL_Fit.Model_spec_per_subj("Sarsa Lambda",
                    RL_Fit.getLL_SarsaLambda_V,
                    sarsaL_prior, sarsaL_prior_std,
                    string(result_path_base, "sarsa_lambda/"))
    # Prioritized Sweeping with Particle Filtering
    # --------------------------------------------
    priosweep_particle_prior = [
        RL_Fit.paramGamma(0.0, 1.0),
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramNrUpdateCycles(0, 9),
        RL_Fit.paramNParticles(1, 20)
        ]
    priosweep_particle_prior_std = [.15, 1.5, .15, 3., 1.35, 3.]
    priosweep_particle_spec = RL_Fit.Model_spec_per_subj("Prioritized Sweeping Particle Filter",
                                RL_Fit.getLL_Prio_Sweep_TParticleFilterJump_V,
                                priosweep_particle_prior,
                                priosweep_particle_prior_std,
                                string(result_path_base, "priosweep_particle/"))
    # Hybrid-λ-PS-PF
    # --------------
    hybrid_pspf_sarsaL_prior = [
        RL_Fit.paramGamma(0.0, 1.0),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramNrUpdateCycles(0, 9),
        RL_Fit.paramNParticles(1, 20),
        RL_Fit.paramAlpha(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.0),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MB"),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MF")
        ]
    hybrid_pspf_sarsaL_prior_std = [.15, .15, 3., 1.35, 3., .15, .15, 3., 3.]
    hybrid_pspf_sarsaL_spec = RL_Fit.Model_spec_per_subj("Hybrid: PS with Particle Filtering + Sarsa Lambda",
                                RL_Fit.getLL_Hybrid_Prio_Sweep_TParticleFilterJump_V,
                                hybrid_pspf_sarsaL_prior, hybrid_pspf_sarsaL_prior_std,
                                string(result_path_base, "hybrid_pspf_sarsaL/"))
    # Actor-Critic
    # ------------
    actorCriticTD_prior = [
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate actor"),
        RL_Fit.paramEta(0.0, 1.; name="learning rate critic"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace actor"), # actor
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace critic"), # critic
        RL_Fit.paramTemp(0.01, 10.)
    ]
    actorCriticTD_prior_std = [.15, .15, .15, .15, .15, 1.5]
    actorCriticTD_spec = RL_Fit.Model_spec_per_subj("Actor-Critic (TD-Lambda)",
                                RL_Fit.getLL_ActorCritic_TDLambda_V,
                                actorCriticTD_prior, actorCriticTD_prior_std,
                                string(result_path_base, "actor_critic_TD/"))
    # REINFORCE
    # ---------
    REINFORCE_prior = [
        RL_Fit.paramAlpha(0.0, 1.),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramTemp(0.01, 10.)
    ]
    REINFORCE_prior_std = [.15, .15, 1.5]
    REINFORCE_spec = RL_Fit.Model_spec_per_subj("REINFORCE (Williams 1992)",
                            RL_Fit.getLL_REINFORCE_V,
                            REINFORCE_prior, REINFORCE_prior_std,
                            string(result_path_base, "REINFORCE/"))
    # Hybrid Actor-Critic
    # -------------------
    hybrid_actorCritic_FWD_prior = [
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate actor"),
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate critic"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace actor"), # actor
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace critic"), # critic
        RL_Fit.paramEta(0., 1.; name="learning rate fwd"),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MB"),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MF")
    ]
    hybrid_actorCritic_FWD_prior_std = [.15, .15, .15, .15, .15, .15, 3., 3.]
    hybrid_actorCritic_FWD_spec = RL_Fit.Model_spec_per_subj("Hybrid Actor-Critic with FWD",
                                RL_Fit.getLL_Hybrid_AC_FWD_V,
                                hybrid_actorCritic_FWD_prior, hybrid_actorCritic_FWD_prior_std,
                                string(result_path_base, "hybrid_actorcritic_FWD/"))

    # Forward Learner (MB only learner)
    # ---------------------------------
    forwardLearner_prior = [
        RL_Fit.paramEta(0., 1.),
        RL_Fit.paramGamma(0., 1.),
        RL_Fit.paramTemp(0.01, 10.)
    ]
    forwardLearner_prior_std = [.15, .15, 1.5]
    forwardLearner_spec = RL_Fit.Model_spec_per_subj("Forward Learner",
                                RL_Fit.getLL_Forward_Learner_V,
                                forwardLearner_prior, forwardLearner_prior_std,
                                string(result_path_base, "forward_learner/"))
    # Hybrid-0 (Glaescher et al. 2010)
    # --------------------------------
    glaescherHybrid_prior = [
        RL_Fit.paramAlpha(0., 1.),
        RL_Fit.paramGamma(0., 1.),
        RL_Fit.paramEta(0., 1.),
        RL_Fit.paramExpDecayOffset(0., 1.),
        RL_Fit.paramExpDecaySlope(-.1, +.8),
        RL_Fit.paramTemp(0.01, 10.)
    ]
    glaescherHybrid_prior_std = [.15, .15, .15, .15, .15, 1.5]
    glaescherHybrid_spec = RL_Fit.Model_spec_per_subj("Glaescher-Hybrid",
                                RL_Fit.getLL_GlaescherHybrid_V,
                                glaescherHybrid_prior, glaescherHybrid_prior_std,
                                string(result_path_base, "glaescher_hybrid/"))
    # Hybrid-λ (Daw et al. 2011)
    # --------------------------
    hybridSarsaL_prior = [
        RL_Fit.paramAlpha(0.0, 1.),
        RL_Fit.paramGamma(0., 1.),
        RL_Fit.paramLambda(0.0, 1.),
        RL_Fit.paramEta(0., 1.),
        RL_Fit.paramExpDecayOffset(0., 1.),
        RL_Fit.paramExpDecaySlope(0., 1.),
        RL_Fit.paramTemp(0.01, 10.)
    ]
    hybridSarsaL_prior_std = [.15, .15, .15, .15, .15, .15, 1.5]
    hybridSarsaL_prior_spec = RL_Fit.Model_spec_per_subj("Hybrid-SarsaLambda",
                                RL_Fit.getLL_HybridSarsaLambda_V,
                                hybridSarsaL_prior, hybridSarsaL_prior_std,
                                string(result_path_base, "hybrid_FWD_SarsaLambda/"))

    # REINFORCE with baseline
    # -----------------------
    REINFORCE_baseline_prior = [
        RL_Fit.paramAlpha(0.0, 1., name="learning rate"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate baseline")
    ]
    REINFORCE_baseline_prior_std = [.15, .15, 1.5, .15]
    REINFORCE_baseline_spec = RL_Fit.Model_spec_per_subj("REINFORCE (Williams 1992) baseline",
                            RL_Fit.getLL_REINFORCE_baseline_V,
                            REINFORCE_baseline_prior, REINFORCE_baseline_prior_std,
                            string(result_path_base, "REINFORCE_baseline/"))
    # Surprise Sarsa-λ
    # ----------------
    sarsaL_modulated_prior = [
        RL_Fit.paramGamma(0.0, 1.0),
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramNParticles(1, 20),
        RL_Fit.paramAlpha(0.0, 1.0),
        RL_Fit.paramLambda(0.0, 1.0)
        ]
    sarsaL_modulated_prior_std = [.15, 1.5, .15, 3., 3., .15, .15]
    sarsaL_modulated_spec = RL_Fit.Model_spec_per_subj("Sarsa Lambda modulated by Particle Filtering",
                                RL_Fit.getLL_sarsaL_modulated_TParticleFilterJump_V,
                                sarsaL_modulated_prior,
                                sarsaL_modulated_prior_std,
                                string(result_path_base, "sarsaL_modulated/"))

    # Surprise Actor-Critic
    # ---------------------
    actorCriticTD_modulatedActor_prior = [
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate actor"),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramEta(0.0, 1.; name="learning rate critic"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace actor"), # actor
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace critic"), # critic
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramNParticles(1, 20)
    ]
    actorCriticTD_modulatedActor_prior_std = [.15, .15, 3., .15, .15, .15, .15, 1.5, 3.]
    actorCriticTD_modulatedActor_spec = RL_Fit.Model_spec_per_subj("Actor-Critic (TD-Lambda) modulated actor (1 degree)",
                                RL_Fit.getLL_ActorCritic_TDLambda_modulatedActor_V,
                                actorCriticTD_modulatedActor_prior, actorCriticTD_modulatedActor_prior_std,
                                string(result_path_base, "actor_critic_TD_modulatedActor_1degree/"))
    # Surprise REINFORCE
    # ------------------
    REINFORCE_modulated_prior = [
        RL_Fit.paramAlpha(0.0, 1.),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramNParticles(1, 20)
    ]
    REINFORCE_modulated_prior_std = [.15, .15, 3., .15, 1.35, 3.]
    REINFORCE_modulated_spec = RL_Fit.Model_spec_per_subj("REINFORCE (Williams 1992) modulated",
                            RL_Fit.getLL_REINFORCE_modulated_V,
                            REINFORCE_modulated_prior, REINFORCE_modulated_prior_std,
                            string(result_path_base, "REINFORCE_modulated/"))

    # Hybrid Actor-Critic PF
    # ----------------------
    hybrid_actorCritic_FWD_BF_prior = [
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate actor"),
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate critic"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace actor"), # actor
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace critic"), # critic
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramNParticles(1, 20),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MB"),
        RL_Fit.paramWeight(0.0, 20.0; name="w_MF")
    ]
    hybrid_actorCritic_FWD_BF_prior_std = [.15, .15, .15, .15, .15, .15, 3., 3., 3., 3.,]
    hybrid_actorCritic_FWD_BF_spec = RL_Fit.Model_spec_per_subj("Hybrid Actor-Critic with FWD - BF",
                                RL_Fit.getLL_Hybrid_AC_FWD_BF_V,
                                hybrid_actorCritic_FWD_BF_prior, hybrid_actorCritic_FWD_BF_prior_std,
                                string(result_path_base, "hybrid_actorcritic_FWD_BF/"))
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --- Run for the paper:
    batch_fit_models_spec = [
                            actorCriticTD_spec,
                            actorCriticTD_modulatedActor_spec,
                            hybrid_actorCritic_FWD_BF_spec,
                            hybrid_actorCritic_FWD_spec,
                            REINFORCE_spec,
                            REINFORCE_modulated_spec,
                            REINFORCE_baseline_spec,
                            sarsaL_spec,
                            sarsaL_modulated_spec,
                            forwardLearner_spec,
                            priosweep_particle_spec,
                            hybridSarsaL_prior_spec,
                            glaescherHybrid_spec,
                            hybrid_pspf_sarsaL_spec,
                            biasedRandomWalk_spec
                          ]
    result_path_base, batch_fit_models_spec
end


# List and specifications of algorithm for parameter recovery
function model_list_paramrecovery(; filestr = "_")

    result_path_base = makehomesavepath("test_main_recovery"*filestr)

    stochmax = 20.

    # Surprise Actor-Critic
    # ---------------------
    actorCriticTD_modulatedActor_prior = [
        RL_Fit.paramAlpha(0.0, 1.; name="learning rate actor"),
        RL_Fit.paramSurpriseProbability(0.0, 1.),
        RL_Fit.paramStochasticity(0.00001, stochmax),
        RL_Fit.paramEta(0.0, 1.; name="learning rate critic"),
        RL_Fit.paramGamma(0.0, 1.),
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace actor"), # actor
        RL_Fit.paramLambda(0.0, 1.; name="elig. trace critic"), # critic
        RL_Fit.paramTemp(0.01, 10.),
        RL_Fit.paramNParticles(1, 20)
    ]
    actorCriticTD_modulatedActor_prior_std = [.15, .15, 3., .15, .15, .15, .15, 1.5, 3.]
    actorCriticTD_modulatedActor_spec = RL_Fit.Model_spec_per_subj("Actor-Critic (TD-Lambda) modulated actor (1 degree)",
                                RL_Fit.getLL_ActorCritic_TDLambda_modulatedActor_V,
                                actorCriticTD_modulatedActor_prior, actorCriticTD_modulatedActor_prior_std,
                                string(result_path_base, "actor_critic_TD_modulatedActor_1degree/"))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --- Run for parameter recovery:
    batch_fit_models_spec = [
                            actorCriticTD_modulatedActor_spec,
                          ]

    result_path_base, batch_fit_models_spec
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------- MCMC settings and helpers --------------------------------------------
# ------------------------------------------------------------------------------

# Settings for data fitting
# Same settings as in runner_fmri_3foldscrossval (see below),
# but here: only fit, not crossvalidation
function runner_fmri_fit(result_path_base,
                                        batch_fit_models_spec
                                        # ; filestr = "_")
                                        )
    rl_module_path = "hi"
    # --------------------------------------------------------------------------
    # ------ Which data? Real or simulated? Edit!

    # Real participants' data:
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEIC_all_fMRI.csv"

    # Simulated data:
    # data_path = "./mcmc_rl_fit/projects/fmri/data_sim/SARSPEICZVG_all_fMRI_Sim_exporder_SurpAC_run2.csv"

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    description = "Batch fit: Models / subjects"
    # ---- DEBUG settings
    max_nr_episodes = 50  # FIT to the first 10 episodes. For the main analysis, I always used 40
    max_nr_episodes = 2  # FIT to the first 10 episodes. For the main analysis, I always used 40
    nr_repetitions = 2
    top_n_fraction = 0.1
    nr_burnin = 3
    nr_samples = 3
    inter_collect_interval = 4
    # # end debug settings
    # ---------------------------
    # ---- Settings used for paper
    # max_nr_episodes = 50  # FIT to the first n episodes.
    # nr_repetitions = 50 #20 # one repetition generates nr_mcmc_samples. doing several repetitions, starting from different random initial positions helps finding isolated regions (e.g. bimodal distributions).
    # top_n_fraction = 0.001 # not important. I sometimes a the top 1percent of mcmc samples
    # nr_burnin = 1500 # important. mcmc needs to "burn in". The more params (larger space) the longer it takes to walk into the volume of interest.
    # nr_samples = 2000 # how many samples to collect from each independent repetition
    # inter_collect_interval = 10 # samples are correlated. therefore make N steps before collecting a sample. 7 to 10 seems a good value.
    # # end
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    fit_spec = RL_Fit.FitSpec_multiple_models(
        batch_fit_models_spec,
        max_nr_episodes,
        nr_burnin,
        nr_samples,
        inter_collect_interval,
        top_n_fraction,
        nr_repetitions,
        # false,  #do_per_subject_fit
        true,  #do_per_subject_fit
        result_path_base
    )
    # --------------------------------------------------------------------------
    # ----------------- RUN -----------------
    RL_Fit.run_fit_multiple_models(description, data_path, fit_spec, rl_module_path)

end


# Settings for crossvalidation
# Same settings as in runner_fmri_fit (see above),
# but here: ONLY crossvalidation, not fit
function runner_fmri_3foldscrossval(result_path_base,
                                        batch_fit_models_spec
                                        )
    rl_module_path = "hi"
    # --------------------------------------------------------------------------
    # ------ Which data? Real or simulated? Edit!

    # Real participants' data:
    data_path = "./mcmc_rl_fit/projects/fmri/data/SARSPEIC_all_fMRI.csv"

    # Simulated data:
    # data_path = "./mcmc_rl_fit/projects/fmri/data_sim/SARSPEICZVG_all_fMRI_Sim_exporder_SurpAC_run2.csv"

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    description = "Batch fit: Models / subjects"
    # ---- DEBUG settings:
    max_nr_episodes = 50  # FIT to the first 10 episodes. For the main analysis, I always used 40
    max_nr_episodes = 2  # FIT to the first 10 episodes. For the main analysis, I always used 40
    nr_repetitions = 2
    top_n_fraction = 0.1
    nr_burnin = 3
    nr_samples = 3
    inter_collect_interval = 4
    # # end debug settings
    # ---------------------------
    # ---- Settings used for paper
    # max_nr_episodes = 50  # FIT to the first n episodes.
    # nr_repetitions = 50 #20 # one repetition generates nr_mcmc_samples. doing several repetitions, starting from different random initial positions helps finding isolated regions (e.g. bimodal distributions).
    # top_n_fraction = 0.001 # not important. I sometimes a the top 1percent of mcmc samples
    # nr_burnin = 1500 # important. mcmc needs to "burn in". The more params (larger space) the longer it takes to walk into the volume of interest.
    # nr_samples = 2000 # how many samples to collect from each independent repetition
    # inter_collect_interval = 10 # samples are correlated. therefore make N steps before collecting a sample. 7 to 10 seems a good value.
    # # end
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    fit_spec = RL_Fit.FitSpec_multiple_models(
        batch_fit_models_spec,
        max_nr_episodes,
        nr_burnin,
        nr_samples,
        inter_collect_interval,
        top_n_fraction,
        nr_repetitions,
        false,  #do_per_subject_fit
        # true,  #do_per_subject_fit
        result_path_base
    )
    # --------------------------------------------------------------------------
    # # ----------------- CROSSVALIDATE -----------------
    (dataFmriAll, all_data_stats) = RL_Fit.load_SARSPEI(data_path; max_episodes_per_subj = max_nr_episodes)
    # when running just an additional model, reuse the same (initially randomly) sampled
    # (make results comparable with previously fit algorithms)

    # I called kf_Spec = RL_Fit.get_k_folds_all_out(Set(all_data_stats.all_subject_ids), 3) once
    # and then fix the subject ID per fold
    kf_Spec = RL_Fit.get_k_folds( Set(all_data_stats.all_subject_ids),
    [
    Set([7, 2, 19, 5, 8, 12, 20])
    Set([18, 4, 13, 3, 15, 21, 6])
    Set([9, 14, 10, 16, 17, 11, 1])])
    # Otherwise shuffle(collect(1:21))... and split in 3
    CV_multi_model_spec = RL_Fit.CV_multiple_models_spec(
        batch_fit_models_spec,
        kf_Spec,
        nr_burnin,
        nr_samples,
        inter_collect_interval,
        top_n_fraction,
        nr_repetitions,
        result_path_base)
    cv_result = RL_Fit.run_cross_validation(description, dataFmriAll,
                                            CV_multi_model_spec, rl_module_path)
end
