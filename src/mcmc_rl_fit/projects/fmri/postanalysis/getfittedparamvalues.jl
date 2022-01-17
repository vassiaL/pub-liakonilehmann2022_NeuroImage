using DataFrames, DataFramesMeta, CSV
using PGFPlots
using Statistics

# Function for results of Fig. A.6
function getfittedparamvalues_multipleruns(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/fits_multipleruns/"),
        filestr = "",
        )

    savepath = makehomesavepath("getfittedparamvalues")
    mkdir(savepath)

    prefix = split(filepath, "/")[end-1]
    runs = getsubfolders(filepath)
    algorithms = getsubfolders(joinpath(filepath, runs[1]))

    for i in algorithms
        dfresults = DataFrame()

        for irun in 1:length(runs)
            if isfile(joinpath(filepath, runs[irun], i, "maxLL_fit/MAP_params.csv"))
                dfalgo = CSV.read(joinpath(filepath, runs[irun], i, "maxLL_fit/MAP_params.csv"), header=false)
                dfresults[:, Symbol("run"*string(irun))] = dfalgo[:, :Column1]
            else
                println("File: $(joinpath(filepath, runs[irun], i, "maxLL_fit/MAP_params.csv")) does not exist!")
            end
        end
        CSV.write(joinpath(savepath, prefix*"_paramvalues_acrossruns_"*i*filestr*".csv"), dfresults,
                delim = " ");
    end
end
