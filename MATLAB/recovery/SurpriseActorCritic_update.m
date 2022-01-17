function learner = SurpriseActorCritic_update(learner, state, action, next_state, reward)
    
    if reward > 0
        done=1;
    else
        done=0;
    end
       
    learner.learnerT = TParticleFilterJump_update(learner.learnerT, state, action, next_state, done);
    modulatingfactor = 1. - learner.learnerT.gamma_surprise;
    
    
    learner = ActorCritic_coreupdate(learner, state, action, next_state, reward, modulatingfactor);
  
   
end