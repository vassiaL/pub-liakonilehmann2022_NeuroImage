# Given the log likelihood (LL) of each seed,
# calculate the log of the mean likelihood,
# i.e. LL = log (sum(LL_seeds) / nseeds)
# INSTEAD OF the mean of the log likelihoods
# i.e. sum (log (LL_seeds)) / nseeds
# Thanks to  Alireza Modirshanechi!
function get_meanLL_acrossseeds(LL_seeds)
    LL_max, seed_idx = findmax(LL_seeds)
    tempsum = 0.
    for i in 1:length(LL_seeds)
        tempsum += exp(LL_seeds[i] - LL_max)
    end
    LL = LL_max + log(tempsum) - log(length(LL_seeds))
end
