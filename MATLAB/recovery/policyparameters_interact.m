function action = policyparameters_interact(learner, state)

    probs = getSoftmaxProbs(learner.policyparameters(state, :), learner.temp);
    
    action = datasample(learner.rgen, 1:2, 1, 'Weights',probs);

end