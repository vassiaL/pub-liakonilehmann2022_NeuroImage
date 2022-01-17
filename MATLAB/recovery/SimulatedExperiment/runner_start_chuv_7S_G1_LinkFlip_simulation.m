function taskResults = runner_start_chuv_7S_G1_LinkFlip_simulation()
% Same as start_chuv_7S_G1_LinkFlip for simulated participants
% Run the simulated experiment with graph 1 of paper
% 2 actions, 7 states

%set the task specific configurations:
taskConfig.Operator = 'vasiliki.liakoni@epfl.ch';
taskConfig.Description = 'fMRI, Tracing RPE and SPE, CHUV -- Sim';
taskConfig.ISI_Min_seconds = 2.0;
taskConfig.ISI_Max_seconds = 7.0;
taskConfig.IEpisodeI_Min_seconds = 2.0;
taskConfig.IEpisodeI_Max_seconds = 7.0;
taskConfig.GoalStateDisplayTime_Min_seconds = 2.5;
taskConfig.GoalStateDisplayTime_Max_seconds = 3.5;
% taskConfig.maxNrOfEpisodes = 100;
taskConfig.maxTotalExperimentDuration_seconds = 60 * 60;

taskConfig.imageFolderName = 'Images_Fractals_normalized_pink';
taskConfig.GoalStateImage = 'HappyWithCup.jpg';
% 
% doEyeTracking = 1;
% dofMRI = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nAgents = 10; % 10 subjects did G1

% Exactly same as experiment:
agentIDsG1 = [4, 5, 6, 8, 10, 12, 15, 16, 19, 21];

% magic = 9;
magic = 12;

% for i=1:nAgents
for i=agentIDsG1 
    
    %%
    taskConfig.maxNrOfEpisodes = randi([43 62], 1); % draw between min and max nr of episodes performed
    
    %%
    taskDef = linkFlipImplementation(taskConfig, @states7_Actions2_G1);

    savepath = what('SimulatedData');
    taskDef.pathname = savepath.path;
    taskDef.filename = strcat('Ssim',num2str(i)); 
    
    %% Agent
%     learner = ActorCritic_init(i);
    learner = SurpriseActorCritic_init(i+magic);
    %%

    taskResults = run_CHUV_circularStimPositions_simulation(taskDef, learner);
end

end