function taskResults = run_CHUV_circularStimPositions_simulation(taskDef, learner)
% Simulated experiment
% Same as run_CHUV_circularStimPositions.m but for simulated subjects.
% No visualization and keyboard interaction.

global GOAL_STATE_ID;

taskResults.StartTime = datestr(clock);
taskResults.taskRunner = mfilename('fullpath');
% sca;

%% Images
% Get folder path to images
% imgFolder = what(taskDef.imageFolderName);
% imgFolder = imgFolder.path;
% Get images
% images = cellfun(@(x) imread(fullfile(imgFolder,x)),taskDef.VisualCueMap, 'UniformOutput',0);
% imagesSizes = cellfun(@(x) size(x),images, 'UniformOutput',0);

% numImages = numel(images);

%% Psychtoolbox set-up
% Make sure we run on OpenGL Psychtoolbox
% AssertOpenGL;
% KbName('UnifyKeyNames');
% Screen('Preference', 'SkipSyncTests', 1)
% % Settings
% if ~exist('doEyeTracking','var')
%     % by default: don't do eye tracking
%     doEyeTracking = 0;
% end
% if ~exist('dofMRI','var')
%     % by default: don't do fMRI
%     dofMRI = 0;
% end
% % if ~exist('fmriDummyScanLag','var')
% %     % by default: don't do fMRI
% %     fmriDummyScanLag = 0;
% % end
% if ~exist('windowSize','var')
%     % by default: show full screen
%     windowSize = [];
% end
% 
% screens = Screen('Screens');
% screenNumber = max(screens);
% backgroundColor = [128 128 128]; % Grey at current screen
% 
% w=Screen('OpenWindow',screenNumber,backgroundColor ,windowSize,32,2, 0);
% 
% % Create the textures from the images
% tex = zeros(1,numImages);
% for stimulusCounter =1:numImages
%     tex(1,stimulusCounter) = Screen('MakeTexture',w,images{stimulusCounter});
% end
% goalImgArray = imread(fullfile(imgFolder,taskDef.GoalStateImage));
% goalImgTexture = Screen('MakeTexture',w, goalImgArray);
% goalImgSize= size(goalImgArray);

% Switch to realtime-priority to reduce timing jitter and interruptions
% caused by other applications and the operating system itself
% Priority(MaxPriority(w));

% Do dummy calls to GetSecs, WaitSecs, KbCheck to make sure
% they are loaded and ready when we need them
%KbCheck;
% WaitSecs(0.1);
% GetSecs;

[resultRows, techIDs, actionNames] = getTechEnums;

eventTimeStampTable = zeros(length(fieldnames(resultRows))+20, 5000);
eventIndex =0;
% end of setup

% KbQueueCreate([]);
% KbQueueStart([]);
% KbEventFlush([]);

% if doEyeTracking
%     % Explicitly enable all keys
%     RestrictKeysForKbCheck([]);
%     el = startEyetrackerEL1000(w, taskDef.edffilename, 0);
% end

%Wait for scanner trigger function
% if dofMRI
%     disp('waitForScannerTrigger...');
%     % Enable only key '5'
%     RestrictKeysForKbCheck(KbName('5%'));
%     triggerTime= waitForScannerTrigger; % wait for scanner to start (the num key 5 will be "pressed")
%     [eventTimeStampTable, eventIndex] = recordActionTimestamp(triggerTime,...
%         actionNames.pseudoActionFmriReferenceTimeZero, actionNames.pseudoActionFmriReferenceTimeZero,...
%         eventTimeStampTable, eventIndex, resultRows);
% end
% % Only get relevant signals, ignore all other key events
% RestrictKeysForKbCheck(getEnabledKeys_2);
% KbEventFlush([]);

% Perform some initial Flip to get us in sync with retrace:

% Clear screen to background color
% Screen('FillRect', w, backgroundColor);
% Screen('Flip', w);

%% Start paradigm
try
    %% Setup done, start the experiment now
%     taksResults.StartTimeStamp = GetSecs;
    taskResults.totalNrOfEpisodes = 0;
%     keyCode=zeros(100,1);
    currentStateId = taskDef.F_SelectStartState();
    expectedStateId=currentStateId;
    isFirstStateOfEpisode=true;
%     subjectResponseTime = 0;
%     goalStateDuration =0;
%     remainingISI=0;
%     isEsc=0;

    while ... %~isEsc ... %exit when escape is pressed
           taskResults.totalNrOfEpisodes<taskDef.maxNrOfEpisodes % or when the maximum number of episode is reached
%             && (GetSecs-taksResults.StartTimeStamp) <= taskDef.maxTotalExperimentDuration_seconds % or when the total time is elapsed
        %% Show white background
%         Screen('FillRect', w, backgroundColor);
        
        % Draw fixation cross
%         drawFixationCross(w, [255,255,255]);
        
        ts = now; %Screen('Flip', w);
        [eventTimeStampTable, eventIndex] = recordIsiTimestamp(ts, eventTimeStampTable, eventIndex, resultRows);
        
        ts = now; %Screen('Flip', w);
           
%         if doEyeTracking
%             Eyelink('Message', 'stim presented, eventIndex=%d', eventIndex+1);
%         end
        [eventTimeStampTable, eventIndex] = recordStateTimestamp(ts,...
            expectedStateId, currentStateId, taskDef.F_GetRewardByStateId(currentStateId),...
            eventTimeStampTable, eventIndex, resultRows);
        
        if currentStateId ~= GOAL_STATE_ID
            
            if isFirstStateOfEpisode  
                eventTimeStampTable(resultRows.startOfEpisode,eventIndex)= 1;
                isFirstStateOfEpisode=false;
            else
                 reward = 0;
                learner = learner.updateFun(learner, previousstate, previousaction, currentStateId, reward);
                
                %% Callback
                eventTimeStampTable(resultRows.RPE_population,eventIndex)= learner.td_error;
                if strcmp(learner.name,'SurpriseActorCritic')
                    eventTimeStampTable(resultRows.SPE_population ,eventIndex)= learner.learnerT.SPE;
                    eventTimeStampTable(resultRows.gamma_surprise_population  ,eventIndex)= learner.learnerT.gamma_surprise;
                    eventTimeStampTable(resultRows.Sbf_population ,eventIndex)= learner.learnerT.Sgm;
                end
                
            end
            
            actionId = learner.interactFun(learner, currentStateId);
            %% ----------
            if actionId==actionNames.undefined
                %do nothing.
                % it is usually the ESCAPE case: we exit this while
                %loop in the next iteration.
                % if it's another button, we simply show the same state
                % once more.
                [eventTimeStampTable, eventIndex] = recordActionTimestamp(ts,...
                    actionNames.undefined, actionNames.undefined, eventTimeStampTable, eventIndex, resultRows);
            else
                previousstate = currentStateId;
                previousaction = actionId;
                [taskDef, currentStateId, effectiveActionId, expectedStateId]= taskDef.F_GetNextStateId(taskDef, currentStateId, actionId);
                [eventTimeStampTable, eventIndex] = recordActionTimestamp(ts,...
                    actionId, effectiveActionId, eventTimeStampTable, eventIndex, resultRows);
            end
            
            
                
        else %currentStateId == GOAL_STATE_ID
            eventTimeStampTable(resultRows.endOfEpisode,eventIndex)= 1;
            
            reward = 1;
            learner = learner.updateFun(learner, previousstate, previousaction, currentStateId, reward);
            
            currentStateId = taskDef.F_SelectStartState();
            expectedStateId=currentStateId;
            
            %end of episode
            taskResults.totalNrOfEpisodes = taskResults.totalNrOfEpisodes+1;
            isFirstStateOfEpisode=true;
            
            %% Callback
                eventTimeStampTable(resultRows.RPE_population,eventIndex)= learner.td_error;
                if strcmp(learner.name,'SurpriseActorCritic')
                    eventTimeStampTable(resultRows.SPE_population ,eventIndex)= learner.learnerT.SPE;
                    eventTimeStampTable(resultRows.gamma_surprise_population  ,eventIndex)= learner.learnerT.gamma_surprise;
                    eventTimeStampTable(resultRows.Sbf_population ,eventIndex)= learner.learnerT.Sgm;
                end
%            
        end
        
    end
%register the exit of the while loop in the eventTable:    
[eventTimeStampTable, eventIndex] = recordActionTimestamp(now, -1, -1, eventTimeStampTable, eventIndex, resultRows);
[eventTimeStampTable, eventIndex] = recordEndOfExperimentTimestamp(eventTimeStampTable, eventIndex, resultRows);    
catch err
%     if doEyeTracking
%         Eyelink('StopRecording');
%         closeEyetrackerEL1000(taskDef.edffilename);
%     end
%     cleanUpPsychotoolbox;
    disp('error in run_CHUV_circularStimPositions_simulation:');
    disp(err.message);
    disp(err.stack);
    nrOfRegisteredEvents = find(eventTimeStampTable(1,:)==0, 1, 'first')-1;
    taskResults.eventTimeStampTable = eventTimeStampTable(:,1:nrOfRegisteredEvents);
    taskResults.EndTime = datestr(clock);
    taskResults.TaskDefinition = taskDef;
    save('./taskResults.mat','taskResults');
    save(fullfile(taskDef.pathname,taskDef.filename),'taskResults');
    rethrow(err);
end
% 
% if doEyeTracking
%     Eyelink('StopRecording');
%     closeEyetrackerEL1000(taskDef.edffilename);
% end
nrOfRegisteredEvents = find(eventTimeStampTable(1,:)==0, 1, 'first')-1;
taskResults.eventTimeStampTable = eventTimeStampTable(:,1:nrOfRegisteredEvents);
taskResults.EndTime = datestr(clock);
taskResults.TaskDefinition = taskDef;
taskResults.learner = learner;
% Save
save(fullfile(taskDef.pathname,taskDef.filename),'taskResults');

save(fullfile(taskDef.pathname,taskDef.filename),'taskResults');
% cleanUpPsychotoolbox;
end

% 
% function cleanUpPsychotoolbox
% %% reset psychotoolbox stuff
% ShowCursor;
% RestrictKeysForKbCheck([]);
% Screen('Close'); % Clear textures
% sca;
% Priority(0);
% KbQueueStop([]);
% KbQueueRelease([]);
% end

function [eventTimeStampTable, eventIndex] = recordStateTimestamp(ts,...
    expectedStateId, stateId, reward,...
    eventTimeStampTable, eventIndex, resultRows)
%writes State information to the eventTimeStampTable
eventIndex=eventIndex+1;
eventTimeStampTable(resultRows.timestamp, eventIndex) = ts;
eventTimeStampTable(resultRows.expectedStateId, eventIndex) = expectedStateId;
eventTimeStampTable(resultRows.stateId, eventIndex) = stateId;
eventTimeStampTable(resultRows.reward, eventIndex) = reward;
end

function [eventTimeStampTable, eventIndex] = recordActionTimestamp(ts,...
    selectedActionId, effectiveActionId,...
    eventTimeStampTable, eventIndex, resultRows)
%writes action Information to the eventTimeStampTable
eventIndex=eventIndex+1;
eventTimeStampTable(resultRows.timestamp, eventIndex) = ts;
eventTimeStampTable(resultRows.selectedActionId, eventIndex) = selectedActionId;
eventTimeStampTable(resultRows.effectiveActionId, eventIndex) = effectiveActionId;
end


function [eventTimeStampTable, eventIndex] = recordIsiTimestamp(ts,...
    eventTimeStampTable, eventIndex, resultRows)
%writes ISI onset Information to the eventTimeStampTable
eventIndex=eventIndex+1;
eventTimeStampTable(resultRows.timestamp, eventIndex) = ts;
eventTimeStampTable(resultRows.IsiOn, eventIndex) = 1;
end

function [eventTimeStampTable, eventIndex] = recordEndOfExperimentTimestamp(...
    eventTimeStampTable, eventIndex, resultRows)
%writes ISI onset Information to the eventTimeStampTable
eventIndex=eventIndex+1;
eventTimeStampTable(resultRows.timestamp, eventIndex) = now;
eventTimeStampTable(resultRows.endOfExperiment, eventIndex) = 1;
end
