function learner = SurpriseActorCritic_init(seed)
    
    learner.name = 'SurpriseActorCritic';
    learner.nr_states = 7; 
    learner.nr_actions = 2;
    learner.alpha_critic = 3.24411e-5;
    learner.alpha_actor = 0.79724; 
    learner.gamma = 0.99664;
    learner.lambda_critic = 0.30497; 
    learner.lambda_actor = 0.79392; 
    learner.temp =  0.39196; 
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
    
    learner.learnerT = TParticleFilterJump_init(seed);
   
    learner.updateFun = @SurpriseActorCritic_update;
    learner.interactFun = @policyparameters_interact;
end