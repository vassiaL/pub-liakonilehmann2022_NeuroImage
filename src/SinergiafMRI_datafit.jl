module SinergiafMRI_datafit

rl_module_path = "./mcmc_rl_fit/src/current_RL_Models.jl"
include(rl_module_path)

include(joinpath("mcmc_rl_fit", "src", "littletools.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "simpleFittingTest.jl"))
include(joinpath("mcmc_rl_fit", "projects", "fmri", "fmri_run_multipletimes.jl"))


include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "behaviour_tools.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "comparesignals.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "getcrossvalresults.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "processreactiontime.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "behaviour_learningcurves.jl"))
include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "behaviour_learningcurves_bootstrap.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "csv_extractevents.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "RPE_SPE_corr.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "behavior_pathlength.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "klaastable.jl"))

include(joinpath("mcmc_rl_fit", "projects", "fmri", "postanalysis", "getfittedparamvalues.jl"))

end # module
