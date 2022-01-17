function learnerT = TParticleFilterJump_init(seed)
% TParticleFilterJump for outlier task (fmri)

% struct TParticleFilterJump <: TPs1a0s0
    learnerT.name = 'PF';
    learnerT.nr_states = 7; 
    learnerT.nr_actions = 2;
    learnerT.nparticles = 15;
    learnerT.Neffthrs = learnerT.nparticles / 2.; 
    learnerT.surpriseprobability = 0.00433;
    learnerT.stochasticity = 2.78163; 
    learnerT.particlesjump = zeros(learnerT.nr_actions, learnerT.nr_states, learnerT.nparticles); 
    learnerT.weights = (1. / learnerT.nparticles) * ones(learnerT.nr_actions, learnerT.nr_states, learnerT.nparticles); 
    
    learnerT.Sgmparticles = zeros(1, learnerT.nparticles);
    learnerT.Sgm = 1.;
    learnerT.m = learnerT.surpriseprobability/ (1. - learnerT.surpriseprobability);
    learnerT.gamma_surprise = learnerT.m/(1. + learnerT.m);
    learnerT.Ps1a0s0 = (1. / learnerT.nr_states) * ones(learnerT.nr_actions, learnerT.nr_states, learnerT.nr_states);
    learnerT.SPE = 1. - 1. / learnerT.nr_states;
    learnerT.counts = zeros(learnerT.nr_actions, learnerT.nr_states, learnerT.nparticles, learnerT.nr_states);
    learnerT.seed = seed;
    learnerT.rgen = RandStream('mt19937ar', 'Seed', seed);
    learnerT.terminalstates = zeros(1, learnerT.nr_states);
   
end