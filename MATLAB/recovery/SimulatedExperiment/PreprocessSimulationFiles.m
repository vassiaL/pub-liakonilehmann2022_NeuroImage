function PreprocessSimulationFiles
% Same as PreprocessFmriFiles.m, but for the simulated agents
% loads all simulated participants' .mat files and saves one 
% preprocessed EventTimeStampTable
% per participant and all participants concatenated 
% 

close all
clear;

csvfilename = 'SARSPEICZVG_all_fMRI_Sim.csv';

currentFolder = what('SimulatedData');
dataFolder = currentFolder.path;

dataFolderContent = what(dataFolder);
allOrigMatFileNames = dataFolderContent.mat;

% Ensure things are overwritter=n
if exist(fullfile(dataFolder, 'preprocessed'), 'dir')
    delete(fullfile(dataFolder, 'preprocessed', '*'))
else
    mkdir(fullfile(dataFolder, 'preprocessed'));
end
cd(fullfile(dataFolder, 'preprocessed'));

nrOfMatFiles = numel(allOrigMatFileNames);
userId_FileMap = cell(nrOfMatFiles,1);

SARSPEICZVG_all = []; % State, Action, Reward, nextState, Puzzle, Episode, Subject, isCatchTrial, reaction time Percentile, Decision Value

for iFile = 1:nrOfMatFiles
% for iFile = 1:2
    fileName = allOrigMatFileNames{iFile};
    userId_FileMap{iFile} = fileName;
    fullFileName = fullfile(dataFolderContent.path, fileName);
    disp(fullFileName);
    load(fullFileName);
    eventTimeStampTable = preprocessEventTimeStampTable(taskResults.eventTimeStampTable);
    %get State-Action-Reward-NextState-Puzzle(=1)-Episode-SubjectID for the
    %current subject:
    SARSECR = getSARSECRfromFrmiEventTable(eventTimeStampTable);
    SARSPEICZVG = [];
    SARSPEICZVG([1,2,3,4,6,8, 9],:) = SARSECR;
    SARSPEICZVG(5,:) = 1;
    SARSPEICZVG(7,:) = iFile;
    
    % Reaction time. To make RT comparable across subjects we compute
    % the per-subject reaction-time percentile
    % --- (NOT USED eventually)
    SARSPEICZVG(9,:) = 100;
    ptile_values = 5:5:95;
    rt_prctiles = prctile(SARSECR(7,:), ptile_values);
    for i_rt = length(rt_prctiles):-1:1
        rt_prctile = rt_prctiles(i_rt);
        idx =find(SARSECR(7,:)<=rt_prctile);
        SARSPEICZVG(9,idx) = ptile_values(i_rt);
    end
%     SARSPEICZVG(12
    
    % assign correct action:
    %% Distinguish between the two graphs
    % --- Graph 1 of paper
    if strcmp(taskResults.TaskDefinition.GraphModel, 'States:7, Action:2, Graphversion: 1')
        
        % stepsToGoal = [0, 0; 3, 2; 2, 3; 4, 1; 2, 2; 1, 3; 3, 3;];
        actionEvaluationMatrix = [...
            [0 0]; ...
            [-1 1]; ...
            [1 -1]; ...
            [-1 1]; ...
            [0 0]; ...
            [1 -1]; ...
            [0 0]; ...
            ];
        goalStateId = 1;
        graphModel = 1;
%         stateLabels = {'G', 'i', 's', 'bG', 'i', 'bG', 's'};
    % --- Graph 2 of paper    
    elseif strcmp(taskResults.TaskDefinition.GraphModel, 'States:7, Action:2, Graphversion: 3')
        
        %stepsToGoal = [0, 0; 1, 2; 1, 3; 3, 2; 4, 2; 3, 2; 3, 3;];
        actionEvaluationMatrix = [...
            [0 0]; ...
            [1 -1]; ...
            [1 -1]; ...
            [-1 1]; ...
            [-1 1]; ...
            [-1 1]; ...
            [0 0]; ...
            ];
        goalStateId = 1;
        graphModel = 3;
%         stateLabels = {'G', 'bG', 'bG', 'i', 'i', 's', 's'};
    else
        fprintf('Unknown task graph model');
    end

    evaluation = cell2mat(arrayfun(@(x,y) actionEvaluationMatrix(x,y), SARSECR(1,:),  SARSECR(2,:), 'UniformOutput', 0));
    SARSPEICZVG(10,:) = evaluation;
    SARSPEICZVG(11,:) = graphModel;
% end correct action    

    SARSPEICZVG_all = horzcat(SARSPEICZVG_all, SARSPEICZVG);
    etsTblFileName = strcat(fileName(1:end-4), '_ETSTbl.mat');
    
    save(etsTblFileName, 'eventTimeStampTable');
    etsTblFileNameCSV = strcat(fileName(1:end-4), '_ETSTbl.csv');
    dlmwrite(etsTblFileNameCSV, eventTimeStampTable, 'precision',9);

end
csvwrite(csvfilename, SARSPEICZVG_all);
save('userId_FileMap.mat','userId_FileMap')
cd(dataFolder)
end

