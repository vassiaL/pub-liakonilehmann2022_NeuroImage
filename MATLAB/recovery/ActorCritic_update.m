function learner = ActorCritic_update(learner, state, action, next_state, reward)

    learner = ActorCritic_coreupdate(learner, state, action, next_state, reward, 1.);
    
end
