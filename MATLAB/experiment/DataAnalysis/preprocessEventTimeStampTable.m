function eventTimeStampTable = preprocessEventTimeStampTable(eventTimeStampTable)
% takes the raw data struct (as saved by the experiment-code).
% adds the following rows to the table:
% 1) resultRows.episodeCount
% 2) resultRows.reactionTime
% 3) resultRows.isCatchTrial

[resultRows, ~, actionNames] = getTechEnums();
% resultRows.episodeCount
eventTimeStampTable(resultRows.episodeCount,:) = cumsum(eventTimeStampTable(resultRows.startOfEpisode,:));
% resultRows.reactionTime; %time from stimulus onset to button press
actionColumns = find(eventTimeStampTable(resultRows.selectedActionId,:)>0);
actionTs = eventTimeStampTable(resultRows.timestamp, actionColumns);
%assert: the stimulusTs is stored just before the actionTs:
statesAtActions = eventTimeStampTable(resultRows.stateId, actionColumns-1);
hasState = prod(statesAtActions)~=0;
if ~hasState
    error('check state and action position in eventTimeStampTable');
end
stimulusTs = eventTimeStampTable(resultRows.timestamp, actionColumns-1);
reactionTime = actionTs-stimulusTs;
eventTimeStampTable(resultRows.reactionTime,actionColumns) = reactionTime;

% resultRows.isCatchTrial aligned with the onset of an unexpected stimulus
unequalStates= find(eventTimeStampTable(resultRows.stateId,:)~=eventTimeStampTable(resultRows.expectedStateId,:));
eventTimeStampTable(resultRows.isCatchTrial,unequalStates)=1;


%get the timestamp relative to each episode:
nrOfEpisodes = max(eventTimeStampTable(resultRows.episodeCount,:));
for e = 1:nrOfEpisodes
    episodeIdxs = find(eventTimeStampTable(resultRows.episodeCount,:)== e);
   tsForEpisode =  eventTimeStampTable(resultRows.timestamp,episodeIdxs);
   tsRelative = tsForEpisode-tsForEpisode(1);
   eventTimeStampTable(resultRows.tsRelativeToEpisode ,episodeIdxs) = tsRelative;
end

%get the timestamp relative to pseudoActionFmriReferenceTimeZero
tsZeroIdx = find(eventTimeStampTable(resultRows.selectedActionId,:)==actionNames.pseudoActionFmriReferenceTimeZero);
if isempty(tsZeroIdx)
    %no fMRI recording. Do not add any additional row
    %(resultRows.tsRelativeToFmriStart)
else
fMRI_tZero = eventTimeStampTable(resultRows.timestamp , tsZeroIdx);
eventTimeStampTable(resultRows.tsRelativeToFmriStart,:) = eventTimeStampTable(resultRows.timestamp,:)-fMRI_tZero;
end
end

