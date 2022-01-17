function learner = ActorCritic_init(seed)
    
    learner.name = 'ActorCritic';
    learner.nr_states = 7; 
    learner.nr_actions = 2;
    learner.alpha_critic = 4.6834e-05;
    learner.alpha_actor = 0.5009; 
    learner.gamma = 0.8206;
    learner.lambda_critic = 0.5783; 
    learner.lambda_actor = 0.9599; 
    learner.temp = 0.2436;
    %
    learner.initvalue = 0.;
    learner.V = zeros(1, learner.nr_states) + learner.initvalue;
    learner.policyparameters = zeros(learner.nr_states, learner.nr_actions);
    learner.eligTrace_critic = zeros(1, learner.nr_states);
    learner.eligTrace_actor = zeros(learner.nr_states, learner.nr_actions);
    learner.td_error = 0.;
    learner.use_replacing_trace = 1;
    learner.do_debug_print = 0;
    learner.seed = seed;
    learner.rgen = RandStream('mt19937ar', 'Seed', seed);
    
    learner.updateFun = @ActorCritic_update;
    learner.interactFun = @policyparameters_interact;

   
end
