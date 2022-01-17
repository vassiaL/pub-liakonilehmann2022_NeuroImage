function getSoftmaxProbs(q_vector, temp, bias_vector; exploration_rate=0.00001)
    # softmax policy n actions
    # adds a bias value to the q values
    biased_q = q_vector .+ bias_vector
    biased_q = biased_q .- maximum(biased_q) # Avoid exp(sth VERY large) = Inf!!!
    nom = exp.(biased_q/temp)
    # nom = checkInf.(nom)
    # if isinf(variable) #|| ratioterm >= 1e16
    denom = sum(nom)
    prob = nom ./ denom
    #avoid some extrema (ln(0)). It's reasonable to assume a very small selection chance
    # even for very big differences in q-values
    # We don't want the LL be dominated by a few extrema (e.g. explorative actions)
    # therefore, shrink each p from [0, 1] to [0.001, 0.999], then normalize to 1
    # println("exploration_rate=$exploration_rate")
    # println("prob=$prob")
    boundedVals = exploration_rate .+ (1.0 - 2*exploration_rate)*prob
    boundedVals = boundedVals ./ sum(boundedVals) # normalize to 1
    # println("boundedVals2: $boundedVals")
end
