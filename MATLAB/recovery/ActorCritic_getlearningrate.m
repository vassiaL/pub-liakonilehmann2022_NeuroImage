function learner = ActorCritic_getlearningrate(learner, modulatingfactor)

    if strcmp(learner.name,'ActorCritic')
        learner.alpha_actor_eff = learner.alpha_actor;
    else
        learner.alpha_actor_eff = learner.alpha_actor * modulatingfactor;
    end
    
end
