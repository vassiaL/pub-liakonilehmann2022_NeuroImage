function taskDef = linkFlipImplementation(taskDef, gridGenerator)
% Core of the task: implements the link flip (surprise) rules. 
% The grid implementation is provided by the gridGenerator parameter (function).

global GOAL_STATE_ID;

taskDef.EnvironmentModel = sprintf('generic Link Flip implementation.');
taskDef.EnvModelImpl = mfilename('fullpath');


%specifies the task
% [state, action] -> new state
[taskDef.StateActionStateTransition,GOAL_STATE_ID,taskDef.StartStateSet, totalNumberOfStates, taskDef.GraphModel] = gridGenerator();
taskDef.initialGoalState = GOAL_STATE_ID;

%randomize the icons used to represent a state
 taskDef.allStateIDs = 1:totalNumberOfStates; 

taskDef.VisualCueMap = getRandomImageNames(totalNumberOfStates, 13); 
%randomize the position at which a state is shown
taskDef.PositionMap = randperm(totalNumberOfStates);

%%%%%%%%%%%%%
taskDef.F_getRandomIsi = @getIsi;
taskDef.F_getRandomGoalStateDisplayTime = @getGoalStateDisplayTime;
taskDef.F_SelectStartState = @selectStartState;
taskDef.F_GetNextStateId = @getNextStateId;
taskDef.F_GetRewardByStateId = @getRewardByStateId;
taskDef.F_getRandomInterEpisodeInterval = @getRandomInterEpisodeInterval;

    function randIsi = getIsi
        randIsi = getUniformFromInterval(taskDef.ISI_Min_seconds, taskDef.ISI_Max_seconds);
    end

    function randIei = getRandomInterEpisodeInterval
        randIei = getUniformFromInterval(taskDef.IEpisodeI_Min_seconds, taskDef.IEpisodeI_Max_seconds);
    end

    function randGoalStateDispTime = getGoalStateDisplayTime
        randGoalStateDispTime = getUniformFromInterval(taskDef.GoalStateDisplayTime_Min_seconds, taskDef.GoalStateDisplayTime_Max_seconds);
    end

    function startStateId = selectStartState
        %select one id from the set
        nrOfStates = length(taskDef.StartStateSet);
        selectedStateIndex = randi(nrOfStates);
        startStateId = taskDef.StartStateSet(selectedStateIndex);
    end

    function reward = getRewardByStateId(stateID)
        if (stateID == GOAL_STATE_ID)
            reward = taskDef.goalStateReward;
        else
            reward = 0;
        end
    end

%the model:
taskDef.goalStateReward = 100;
[nrOfStates,nrOfActions] = size(taskDef.StateActionStateTransition);

taskDef.TransitionMatrix = (1/nrOfStates) * ones(nrOfStates, nrOfActions, nrOfStates);
taskDef.QsaTable = zeros(nrOfStates, nrOfActions);
taskDef.Vtable = zeros(nrOfStates,1);

%% define the catch-trial parameters:
% don't do catch-trials during the first n episodes:

taskDef.nrOfInitialNoCatchEpisodes = 4;
taskDef.noCatch_LowerBound = 3;
taskDef.noCatch_UpperBound = 8;
taskDef.randomCatchTrialRate = 0.07;

%define the allowed difference in V-Value
taskDef.vValueMargin = 0.25;
% only do catch-trials on strongly learned transitions
taskDef.transitionProbabilityPercentile = 70;

%for SARSA: keep track of the previous state/action;
taskDef.previousState =NaN;
taskDef.previousAction =NaN;


%tech counters:
taskDef.episodeCount =0;
taskDef.stepsSinceLastCatchTrial = 0;

end

% function randomNumber = getUniformFromInterval(lowerBound, upperBound)
% %draws a random number from a uniform distribution in [lowerBound, upperBound]
% randomNumber = rand;
% intervalLength = upperBound - lowerBound;
% randomNumber = randomNumber*intervalLength;
% randomNumber = randomNumber+lowerBound;
% end


function [modifiedTaskDef, nextStateId, effectiveActionId, expectedNextState] = getNextStateId(taskDefinition, currentStateId, selectedActionId)
%deterministic case: use the selected action.
global GOAL_STATE_ID;
modelLearningRate_eta =0.25;
sarsaDiscountRate_gamma = 0.5;
sarsaLearningRate_alpha = 0.5;

effectiveActionId= selectedActionId;
expectedNextState = taskDefinition.StateActionStateTransition(currentStateId, effectiveActionId);


%% catch trial:
fprintf('taskDefinition.episodeCount: %d\n', taskDefinition.episodeCount);

isCatchTrial = false;

if (taskDefinition.episodeCount >= taskDefinition.nrOfInitialNoCatchEpisodes)
    %check for lower bound:
    if (taskDefinition.stepsSinceLastCatchTrial >= taskDefinition.noCatch_LowerBound)
        
        if (rand < taskDefinition.randomCatchTrialRate)
            % add some random jumps and get some variance ...
            candidates =  taskDefinition.allStateIDs(taskDefinition.allStateIDs~=expectedNextState & taskDefinition.allStateIDs~=GOAL_STATE_ID & taskDefinition.allStateIDs~=currentStateId);
            nrOfCand = length(candidates);
            nextStateId = candidates(randi(nrOfCand));
            isCatchTrial = true;
            %                             beep = MakeBeep(880,0.1);
            %                 Snd('Open');
            %                 Snd('Play',beep);
            
        else
            tmVector = reshape(taskDefinition.TransitionMatrix, 1, numel(taskDefinition.TransitionMatrix));
            transitionProbabilityThreshold = prctile_modified(unique(tmVector), taskDefinition.transitionProbabilityPercentile);
            transitionProbability = taskDefinition.TransitionMatrix(currentStateId, selectedActionId,expectedNextState);
            
            fprintf('StateId: %d, ActionId: %d, NextStateId: %d\n',currentStateId, selectedActionId, expectedNextState);
            fprintf('transitionProbabilityThreshold: %0.2f, transitionProbability: %0.2f\n)',transitionProbabilityThreshold, transitionProbability);
            if (transitionProbability >= transitionProbabilityThreshold)
                disp('check for catch trial')
                vNextState=taskDefinition.Vtable(expectedNextState)
                vValueRange = taskDefinition.vValueMargin * vNextState;
                absVvalueDiff= abs(taskDefinition.Vtable-vNextState);
                candidates = find(absVvalueDiff <= vValueRange);
                candidates = candidates(candidates~=expectedNextState & candidates~=GOAL_STATE_ID & candidates~=currentStateId);
                if (isempty(candidates))
                    disp('no candidate found -> no catch trial')
                    nextStateId = taskDefinition.StateActionStateTransition(currentStateId, effectiveActionId)
                else
                    disp('jump!');
                    
                    %debug: beep
                    %                     beep = MakeBeep(600,0.1);
                    %                     Snd('Open');
                    %                     Snd('Play',beep);
                    
                    isCatchTrial = true;
                    nrOfCand = length(candidates);
                    nextStateId = candidates(randi(nrOfCand));
                end
            else
                disp('no catch trial');
                nextStateId = taskDefinition.StateActionStateTransition(currentStateId, effectiveActionId);
            end
            
            if (~isCatchTrial) & (taskDefinition.noCatch_UpperBound <= taskDefinition.stepsSinceLastCatchTrial)
                %check the upper bound and enforce a catch: we want to avoid the (unlikely)
                %case of very long sequences without any surprise signal.
                %Debug: beep
                %                 beep = MakeBeep(1200,0.1);
                %                 Snd('Open');
                %                 Snd('Play',beep);
                disp('enforce jump');
                candidates =  taskDefinition.allStateIDs(taskDefinition.allStateIDs~=expectedNextState & taskDefinition.allStateIDs~=GOAL_STATE_ID & taskDefinition.allStateIDs~=currentStateId);
                nrOfCand = length(candidates);
                nextStateId = candidates(randi(nrOfCand));
                isCatchTrial = true;
            end
        end
    else
        disp('no catch trial because lower bound not met.');
        nextStateId = taskDefinition.StateActionStateTransition(currentStateId, effectiveActionId);
    end % checked for nrOfSteps since last catch trial
    
    
else
    disp('no catch episode');
    nextStateId = taskDefinition.StateActionStateTransition(currentStateId, effectiveActionId)
end %checked for first episodes: those don't have a catch trial


if isCatchTrial | (taskDefinition.episodeCount < taskDefinition.nrOfInitialNoCatchEpisodes)
    taskDefinition.stepsSinceLastCatchTrial =0;
    %taskDefinition.catchTrial = 1;
else
    taskDefinition.stepsSinceLastCatchTrial = taskDefinition.stepsSinceLastCatchTrial+1;
    %taskDefinition.catchTrial = 0;
end



%% keep track of the transitions and state/action pairs--> compute SARSAs Q(s,a) values
statePredictionError = 1-taskDefinition.TransitionMatrix(currentStateId, selectedActionId,nextStateId);
%taskDefinition.SPE = statePredictionError;
fprintf('statePredictionError: %0.2f\n', statePredictionError);
newTransitionProbability = taskDefinition.TransitionMatrix(currentStateId, selectedActionId,nextStateId) + modelLearningRate_eta*statePredictionError;

%decrease the transition probabilities of the other s'
taskDefinition.TransitionMatrix(currentStateId, selectedActionId,:)=...
    (1-modelLearningRate_eta) * taskDefinition.TransitionMatrix(currentStateId, selectedActionId,:);

taskDefinition.TransitionMatrix(currentStateId, selectedActionId,nextStateId)= newTransitionProbability;


%% SARSA: compute the Q(s,a) table on the fly: we need this information to
%implement State-Swaps dynamically.
if (isnan(taskDefinition.previousState))
    %this is the first (s,a) in an episode. Nothing to do here
    %taskDefinition.RPE(1) = 0;
else
    reward = taskDefinition.F_GetRewardByStateId(currentStateId);
    rewardPredictionError = reward  ...
        + sarsaDiscountRate_gamma*taskDefinition.QsaTable(currentStateId, selectedActionId) ...
        - taskDefinition.QsaTable(taskDefinition.previousState, taskDefinition.previousAction);
    taskDefinition.QsaTable(taskDefinition.previousState, taskDefinition.previousAction) = ...
        taskDefinition.QsaTable(taskDefinition.previousState, taskDefinition.previousAction) ...
        + sarsaLearningRate_alpha * rewardPredictionError;
    taskDefinition.Vtable(taskDefinition.previousState) =  computeVs(taskDefinition.QsaTable(taskDefinition.previousState, :));
    fprintf('rewardPredictionError: %0.2f\n', rewardPredictionError);
    %taskDefinition.RPE(1) = rewardPredictionError;
end

%check for goal state.
if (nextStateId == GOAL_STATE_ID)
    %goal state: update the Q(s,a) value for the current (s,a): There is
    %no (s_t+1, a_t+1);
    reward = taskDefinition.F_GetRewardByStateId(nextStateId);
    rewardPredictionError = reward  ...
        - taskDefinition.QsaTable(currentStateId, selectedActionId);
    taskDefinition.QsaTable(currentStateId, selectedActionId) = ...
        taskDefinition.QsaTable(currentStateId, selectedActionId) ...
        + sarsaLearningRate_alpha * rewardPredictionError;
    taskDefinition.Vtable(currentStateId) =  computeVs(taskDefinition.QsaTable(currentStateId, :));
    
    taskDefinition.episodeCount = taskDefinition.episodeCount+1;
    taskDefinition.previousState =NaN;
    taskDefinition.previousAction =NaN;
    fprintf('rewardPredictionError: %0.2f\n', rewardPredictionError);
    %taskDefinition.RPE(2) = rewardPredictionError;
else
    taskDefinition.previousState =currentStateId;
    taskDefinition.previousAction =selectedActionId;
    %taskDefinition.RPE(2) = 0;
end

modifiedTaskDef = taskDefinition;

end

function Vs = computeVs(Qsa)
%Compute the V value for a state from the Q(s,a) of that same state.
% Decision: take the maximum.
%Review this if you have a better idea...
Vs = max(Qsa);

end


