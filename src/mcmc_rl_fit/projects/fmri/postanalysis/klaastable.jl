using DataFrames, DataFramesMeta, CSV
using PGFPlots
using Statistics
using DelimitedFiles

# Create a table of log-evidence x subject
# in order to perform a random effects analysis (Rigoux et al., 2014; Stephan et al., 2009)
# in MATLAB
# Like the one that now is in someresults/KlaasAnalysis/
function createklaastable_crossvalLL(;
        filepath =  joinpath(dirname(pathof(SinergiafMRI_datafit)),"mcmc_rl_fit/projects/fmri/someresults/crossvalidation/"),
        filestr = "",
        )

    savepath = makehomesavepath("createklaastable")
    mkdir(savepath)

    nSubj = 21
    prefix = split(filepath, "/")[end-1]
    runs = getsubfolders(filepath)
    algorithms = getsubfolders(joinpath(filepath, runs[1]))

    LLallalgos = []
    LLalgo = []
    LLallalgosmean = zeros(nSubj, length(algorithms))
    LLallalgosstd = zeros(nSubj, length(algorithms))
    LLallalgosstderror = zeros(nSubj, length(algorithms))
    for i in 1:length(algorithms)

        for irun in 1:length(runs)

            if irun == 1 # Initialize
                 LLalgo = zeros(length(runs), nSubj)
            end

            dfalgo = CSV.read(joinpath(filepath, runs[irun], algorithms[i],
                    "cross_validation/LL_out_subjectwise.csv"), header=false)
            LLalgo[irun, :] = [sum(dfalgo[:,j]) for j in 1:size(dfalgo,2)]

        end
        push!(LLallalgos, LLalgo)
        LLallalgosmean[:, i] = mean(LLalgo, dims=1)
        LLallalgosstd[:, i] = std(LLalgo, dims=1)
        LLallalgosstderror[:, i] = std(LLalgo, dims=1) ./sqrt(length(runs))
    end
    open(joinpath(savepath, "LLmean_algoxsubj.csv"), "a") do io
       writedlm(io, transpose(LLallalgosmean))
       end
    open(joinpath(savepath, "LLstd_algoxsubj.csv"), "a") do io
       writedlm(io, transpose(LLallalgosstd))
       end
    open(joinpath(savepath, "LLstderror_algoxsubj.csv"), "a") do io
        writedlm(io, transpose(LLallalgosstderror))
        end
    open(joinpath(savepath, "algolist.csv"), "a") do io
       writedlm(io, algorithms)
       end

    LLallalgosmean, LLallalgosstd, LLallalgosstderror# LLallalgos
end
