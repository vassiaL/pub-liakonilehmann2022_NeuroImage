function taskResults = start_chuv_7S_G3_LinkFlip
% Run the experiment with graph 2 of paper
% 2 actions, 7 states

%set the task specific configurations:
taskConfig.Operator = 'vasiliki.liakoni@epfl.ch';
taskConfig.Description = 'fMRI, Tracing RPE and SPE, Brainstorming';
taskConfig.ISI_Min_seconds = 2.0;
taskConfig.ISI_Max_seconds = 7.0;
taskConfig.IEpisodeI_Min_seconds = 2.0;
taskConfig.IEpisodeI_Max_seconds = 2.0;
taskConfig.GoalStateDisplayTime_Min_seconds = 2.5;
taskConfig.GoalStateDisplayTime_Max_seconds = 3.5;
taskConfig.maxNrOfEpisodes = 100;
taskConfig.maxTotalExperimentDuration_seconds = 15 * 60;

taskConfig.imageFolderName = 'Images_Fractals_normalized_pink';
taskConfig.GoalStateImage = 'HappyWithCup.jpg';

doEyeTracking = 0;
dofMRI = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
taskDef = linkFlipImplementation(taskConfig, @states7_Actions2_G3);

[taskDef.setUp, taskDef.filename, taskDef.pathname] = expSetUp;
taskDef.edffilename = strcat('S',taskDef.setUp.subjId, ...
    'R',taskDef.setUp.runId);

debugWindowSize = [];

taskResults = run_CHUV_circularStimPositions(taskDef, doEyeTracking, dofMRI, debugWindowSize);

end