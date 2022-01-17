RL algorithms
-------------
The module 'current_RL_Models.jl' is loaded.
Within that file, a couple of other julia files are imported as well as the file 'discrete_RL_replay', which includes all the RL algorithms implemented for the paper.

For each algorithm there is code at three locations: (e.g for Actor-critic)
1. In src/mcmc_rl_fit/src/RL_impl/ (e.g actor_critic.jl).
The corresponding file contains the application of the algorithm on the participants' data. 
It returns:
    a) the resultings LogLikelihood values (**maxLL**) for the given data and parameters, and ,
    b) some results and metadata (**signals**), e.g RPE, SPE, etc (in the Step_by_step_signals struct). Signals other than RPE or SPE are stored in the **results_dict** in signals with a new key=>value pair.

2. In src/mcmc_rl_fit/src/RL_impl/learner/ (e.g actorcriticlearner.jl)
Here are the core functions of the algorithms' implementation. 

3. For each algorithm there is a function in the file src/mcmc_rl_fit/src/current_RL_Models.jl
    (e.g. getLL_ActorCritic_TDLambda_V).
    This is a wrapper (calling the actor_critic.jl) that makes it callable from within the MCMC framework.

Writing of output results
-------------------------
For each algorithm a new folder is created (in your home directory with the current timestamp).
All the per-subject, per-trial signals (SPE, RPE) are stored there. 
The function _write_subj_data_file() in src/mcmc_rl_fit/src/sinergia_fit_tools.jl writes the contents to the files,
For any new key=>value to the signals structure (as mentioned above), you also need to make sure it is written to a file in _write_subj_data_file().
Once the learning signals (RPE, SPE, others) are written to these per-participand files, they can be copied back into a matlab structure (EventTimeStampTable) via the file /MATLAB/experiment/DataAnalysis/add_SPE_RPE_to_EventTable.m.

All functions for data analysis and for the paper's results (in /src/mcmc_rl_fit/projects/fmri/postanalysis) save their outputs in newly created folders in your home directory with a prefix that corresponds to each function and the current timestamp.


Model fitting and model comparison
----------------------------------
To run the model fitting and model comparison (crossvalidation) procedure of the paper, run the function runner_crossval_multipletimes_fit() in src/mcmc_rl_fit/projects/fmri/fmri_run_multipletimes.jl

These procedures take a long time to run.
The settings are different from the ones used in the paper (DEBUG settings).
Comment/Uncomment the corresponding lines (around 383 and 445)
to perform the fitting and comparison procedures as in the paper.

For these analyses the real participants' data (/src/mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv) are loaded.
This file is one large matrix of participants' data concatenated.
See /MATLAB/readme.txt for more details on participants' data recording and conventions.


Model and parameter recovery
----------------------------
Run runner_multipletimes_fit_recovery() in src/mcmc_rl_fit/projects/fmri/fmri_run_multipletimes.jl

Right now the real participants' data are loaded.
Comment/Uncomment the corresponding lines (around 367 and 428)
to use the simulated data for model and parameter recovery.

As above, comment/Uncomment lines around 383 and 445 to edit the settings used.


Behavioral data analysis
------------------------
The folder src/mcmc_rl_fit/projects/fmri/postanalysis contains functions used to analyze the data and obtain the results of the paper's figures.
These functions load some preprocessed data or results (e.g output of crossvalidation procedure) from the folder src/mcmc_rl_fit/projects/fmri/someresults


Figures
-------
The .csv files in /data/ are then used by the .tex scripts in /figs/ to create the paper's figures.
You can reproduce the figures by running these .tex files.
We used pdfTeX, Version 3.14159265-2.6-1.40.20 (TeX Live 2019) (preloaded format=pdflatex 2019.5.24)


Note
----
We are unfortunately not allowed to share the brain imaging data of the participants.


