function [learner, next_action] = interact(learner, state, action, next_state, reward)
     
    learner = learner.updateFun(learner, state, action, next_state, reward);
    
  
end