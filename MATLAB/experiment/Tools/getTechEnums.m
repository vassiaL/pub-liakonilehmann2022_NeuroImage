function [resultRows, techIDs, actionNames, eventFlags] = getTechEnums
% Enum-like definitions
%
% During the experiment, data are recorded in a table 
% (called 'eventTimeStampTable').
% Each row in the eventTimeStampTable corresponds to an event or a quantity
% defined here.
%
% For example the 1st row records the time of each event, 
% the 2nd row indicates that start of an interstimulus interval, etc...
%
% Some rows defined here are blank at data acquisition and were used for
% later data analysis steps 
% (e.g. resultRows.PolicyParametersDiffChosenVsUnchosen_population).
% 

%setup 
resultRows.timestamp =1;
resultRows.IsiOn = 2;
resultRows.selectedActionId = 3;
resultRows.effectiveActionId = 4;
resultRows.expectedStateId=5;
resultRows.stateId=6;
resultRows.reward = 7;
resultRows.startOfEpisode = 8;
resultRows.endOfEpisode = 9;
resultRows.endOfExperiment = 10; %end of experiment or ESC

%rows added by post processing/data analysis
resultRows.episodeCount = 11; % the episode to which this state/action/... belongs to.
resultRows.tsRelativeToEpisode = 12; % ts(row 1) is absolute. this ts is set to 0 at every episode. used to align signals from Pupillometry
resultRows.reactionTime = 13; %time from stimulus onset to button press
resultRows.isCatchTrial = 14; %aligned with the onset of an unexpected stimulus

%currently unused
% resultRows.isOldGoalState = 15; % aligned with the (first) onset of a state that was the goal in previous goal episodes
% resultRows.isNewGoalState = 16; % aligned with the first reach of a new goal state. including the first goal state of the first episode.

resultRows.tsRelativeToFmriStart = 17; %aligned with pseudoActionFmriReferenceTimeZero
resultRows.eventFlags = 18; 
resultRows.puzzleCount = 19;
resultRows.actionIsCorrect = 20; %1 if the action follows the shortest path.
resultRows.stateVisitCount = 21; %nr of times a state was seen. Refers to S in S,a,s' and is aligned with resultRows.stateId
resultRows.stateActionCount = 22; %nr of times a specific action a is taken in state s (regardless of the outcome). Is aligned with resultRows.selectedActionId
resultRows.episodeCountPerPuzzle = 23; %episode counter that starts at 1 for each puzzle
resultRows.stateCount = 24; % a counter, incremented each time a state is shown. reset per puzzle. Helps to distinguish the case when (e.g.) S3 is visited the first time vs. the case when S3 is the first (or second or...) state seen in a puzzle
resultRows.RPE_population = 25; % Reward Prediction Error using population parameters
resultRows.RPE_subject = 26; % Reward Prediction Error using per-subject parameters
resultRows.SPE_population = 27; % State Prediction Error using population parameters
resultRows.SPE_subject = 28; % State Prediction Error using per-subject parameters

resultRows.gamma_surprise_population = 29; % gamma Surprise using population parameters
resultRows.gamma_surprise_subject = 30; % gamma Surprise using per-subject parameters

resultRows.action_selection_prob_population = 31;
resultRows.action_selection_prob_subject = 32;

resultRows.Sbf_population = 33; % Surprise using population parameters
resultRows.Sbf_subject = 34; % Surprise using per-subject parameters

resultRows.DeltaPolicy_population = 35; % REINFORCE Delta policy parameter (absolute)
resultRows.DeltaPolicy_subject = 36; % REINFORCE Delta policy parameter (absolute)

resultRows.StateValue_population = 37; % V Value
resultRows.StateValue_subject = 38; % V Value

resultRows.PolicyParametersDiffChosenVsUnchosen_population = 39; % p(s,a_chosen) - p(s,a_notchosen)
resultRows.PolicyParametersDiffChosenVsUnchosen_subject = 40; % p(s,a_chosen) - p(s,a_notchosen)



techIDs.StartButtonPseudoStateId = -1;

actionNames.pseudoActionFmriReferenceTimeZero = -5;
%actionNames.pseudoActionFmriStart = -4; %not used anymore
actionNames.pseudoActionStartButton = -3;
actionNames.pseudoActionTimeout = -2;
actionNames.undefined =-1;
actionNames.up = 1;
actionNames.right = 2;
actionNames.down = 3;
actionNames.left = 4;


eventFlags.cueFinished = 100;
eventFlags.eyeTrackerOn = 101;
eventFlags.eyeTrackerOff = 102;

%sounds or visual cues are played at the beginning of each puzzle
eventFlags.PreplayStart = 201;
eventFlags.PreplayEnd = 202;
eventFlags.PreplaySoundOn = 203;
eventFlags.PreplaySoundOff = 204;
eventFlags.PreplayVisualCueOn = 205;
eventFlags.PreplayVisualCueOff = 206;

%messages are shown on the screen
eventFlags.msgOn_NewPuzzleGenerated  = 500;
eventFlags.msgOff_NewPuzzleGenerated  = 501;
eventFlags.msgOn_PuzzleStartsNow = 502;
eventFlags.msgOff_PuzzleStartsNow = 503;
eventFlags.msgOn_AllSoundsArePlayedNow = 504;
eventFlags.msgOff_AllSoundsArePlayedNow = 505;
eventFlags.msgOn_ExperimentStarts = 506;
eventFlags.msgOff_ExperimentStarts = 507;



end

