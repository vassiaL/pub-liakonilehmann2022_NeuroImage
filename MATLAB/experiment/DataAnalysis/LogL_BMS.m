%% Load Data
filepath = '../../../src/mcmc_rl_fit/projects/fmri/someresults/KlaasAnalysis/LLmean_algoxsubj.csv';
Data = csvread(filepath);
Data = Data';

nSubj = size(Data,1);
nAlgo = size(Data,2);

savepath = '../../temp/';

%% ------ Winning algorithms only:
% REINFORCE, Actor-critic, Surprise Actor-critic, Hybrid Actor-critic, 
% Surprise REINFORCE, REINFORCE baseline, Hybrid Actor-critic BF
selection_PG = [1,2,3,4,5,10,11];
Data_PG = Data(:,selection_PG);
% Save Uniform
[Benchmark_alpha_PG,Benchmark_exp_r_PG,Benchmark_xp_PG,Benchmark_pxp_PG,Benchmark_bor_PG] = spm_BMS(Data_PG,[],[],[],[],ones(1,length(selection_PG)));
dlmwrite(fullfile(savepath, 'alpha_PG.csv'), Benchmark_alpha_PG, 'delimiter',' ');
dlmwrite(fullfile(savepath, 'exp_r_PG.csv'), Benchmark_exp_r_PG, 'delimiter',' ');
dlmwrite(fullfile(savepath, 'xp_PG.csv'), Benchmark_xp_PG, 'delimiter',' ');
dlmwrite(fullfile(savepath, 'pxp_PG.csv'), Benchmark_pxp_PG, 'delimiter',' ');
dlmwrite(fullfile(savepath, 'bor_PG.csv'), Benchmark_bor_PG, 'delimiter',' ');
dlmwrite(fullfile(savepath, 'selection_PG.csv'), selection_PG, 'delimiter',' ');