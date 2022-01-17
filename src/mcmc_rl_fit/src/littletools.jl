# Helper functions

using Dates

function makehomesavepath(foldername)
    rightnow = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

    savepath = joinpath(homedir(), foldername * "_" * rightnow * "/")

    while isdir(savepath)
        sleep(1.)
        rightnow = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

        savepath = joinpath(homedir(), foldername * "_" * rightnow * "/")
    end
    savepath
end

function checkifpathexists(result_path_base)
    if ispath(result_path_base)
        if "del" in ARGS
            println("rm($result_path_base)")
            rm(result_path_base; recursive=true)
        elseif "append" in ARGS
        # nothing to do, append new content to the result folder
        else
            error("Result folder already exists: $result_path_base")
        end
    end
end

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,dims) = mapslices(nanmean, x, dims=dims)
nanstd(x) = std(filter(!isnan,x))
nanstd(x,dims) = mapslices(nanstd, x, dims=dims)
nanlength(x) = length(filter(!isnan,x))
nanlength(x,dims) = mapslices(nanlength, x, dims=dims)
function nanserror(x, dims)
    std = nanstd(x, dims)
    n = nanlength(x, dims)
    std ./ sqrt.(n)
end
function calc_stats_nan(dataarray) # dataarray : timepoints x instances

    m = vec(nanmean(dataarray, 2))
    s = vec(nanstd(dataarray, 2))
    serror = vec(nanserror(dataarray, 2))
    m, s, serror
end

function checkInf(variable)
    if isinf(variable) #|| ratioterm >= 1e16
        #println("I'm Inf")
        variable = 1e16
    end
    variable
end


searchdirendings(path, key) = filter(x->endswith(x, key), readdir(path))

getsubfolders(path) = [x for x in readdir(path) if isdir(joinpath(path,x))]

# Dataframe manipulations
function dftranspose(;
                    filepath =  "../R2adj/",
                    filename = "medianR2adj.csv",
                    titleRowsSymbols = [:REINFORCE, :AC, :SurpAC, :HybridAC, :SurpREINFORCE],
                    savepath = "./",
                    )

    df = CSV.read(joinpath(filepath, filename), header=false)

    dftrans = dfjusttranspose(df)

    dftrans = dfplacetitlesinfirstcolumn(dftrans, titleRowsSymbols=titleRowsSymbols)

    prefix = split(filename, ".")[end-1]
    CSV.write(joinpath(savepath, prefix * "_trans.csv"), dftrans, delim = " ")
    dftrans
end

function dfjusttranspose(df)
    dftrans = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
end


function dfstatspercolumn(;
                filepath =  "../R2adj_fitruns/",
                filename = "medianR2adj.csv",
                titleRowsSymbols = [:run1, :run2, :run3, :run4, :run5, :run6],
                savepath = "./",
                )

    df = CSV.read(joinpath(filepath, filename), header=false)

    dfstats = dfjustdostats(df, titleRowsSymbols)

    prefix = split(filename, ".")[end-1]
    CSV.write(joinpath(savepath, prefix * "_statistics.csv"), dfstats, delim = " ")
    dfstats
end

function dfjustdostats(df, titleRowsSymbols)

    dfstats = DataFrame(
                column = titleRowsSymbols,
                _mean =  mean.(eachcol(df)),
                _std = std.(eachcol(df)),
                _serror = std.(eachcol(df))./sqrt.(length.(eachcol(df)))
                )
    dfstats
end

function dfplacetitlesinfirstcolumn(df;
                    titleRowsSymbols = [:REINFORCE, :AC, :SurpAC, :HybridAC]
                    )

    if ~isempty(titleRowsSymbols)
        if length(titleRowsSymbols) == size(df,1)
            df[:,1] = titleRowsSymbols
        else
            println("Length of row titles does not match!")
        end
    else
        println("No row titles")
    end
    df
end


function dfsort(df)

    dfstats = DataFrame(meanmedR2adj =  mean.(eachcol(df)))

    p = sortperm(dfstats[:,1])

     dsorted = df[:,p]
 end

function dfcenterwrtlastcolumn(df)

    dfcentered = df .- df[:,end]

end

# deleterows!(dftrans,1)
# rename!(dftrans, titleRowsSymbols)
