function learner = ActorCritic_coreupdate(learner, state, action, next_state, reward, modulatingfactor)

    probs = getSoftmaxProbs(learner.policyparameters(state, :), learner.temp);
    p_action = probs(action);
    
%     # @show p_action
    if learner.use_replacing_trace % replacing trace
        learner.eligTrace_critic(state) = 1.;
        learner.eligTrace_actor(state, action) = (1. - p_action); %see eq. 7.12/7.13 in Sutton & Barto
    else % (accumulating) eligibility trace
        learner.eligTrace(state) = learner.eligTrace(state) + 1.;
        learner.eligTrace_actor(state, action) = learner.eligTrace_actor(state, action) + (1. - p_action); %see eq. 7.12 in Sutton & Barto
    end

    learner = ActorCritic_getlearningrate(learner, modulatingfactor);

    learner.td_error = reward + (learner.gamma * learner.V(next_state)) - learner.V(state);

    learner.V = learner.V + learner.alpha_critic  * learner.td_error * learner.eligTrace_critic;
    learner.policyparameters = learner.policyparameters + learner.alpha_actor_eff * learner.td_error * learner.eligTrace_actor;
    
    if reward>0
        learner.eligTrace_critic = zeros(1, learner.nr_states);
        learner.eligTrace_actor = zeros(learner.nr_states, learner.nr_actions);
    else % non goal state
        learner.eligTrace_actor = learner.eligTrace_actor * learner.lambda_actor * learner.gamma;
        learner.eligTrace_critic = learner.eligTrace_critic * learner.lambda_critic * learner.gamma;
    end
    
end
