function [ SARSECR ] = getSARSECRfromFrmiEventTable( eventTimeStampTable)
% Helper function for processing participants' .mat files

goalStateId = eventTimeStampTable(6,find(eventTimeStampTable(7,:)>0, 1)); %find first reward, get goalStateId
referenceIdx = find(eventTimeStampTable(6,:)>0 & eventTimeStampTable(6,:)~=goalStateId);
states = eventTimeStampTable(6, referenceIdx);
action = eventTimeStampTable(3, referenceIdx+1);
reaction_time = eventTimeStampTable(13, referenceIdx+1);
episode = eventTimeStampTable(11, referenceIdx);
isCatchTrial = eventTimeStampTable(14, referenceIdx);

%check whether there's a next state (or if the experiment was stopped)
lastIdxNextState = referenceIdx(end)+3;
nrOfSarsaTuples = numel(states);
[m,n]= size(eventTimeStampTable);
if (lastIdxNextState>n)
    referenceIdx(end) = []; %remove last idx
    %also get rid of incomplete / cancelled state/actions:
    nrOfSarsaTuples = nrOfSarsaTuples-1;
end

nextState = eventTimeStampTable(6, referenceIdx+3);
reward = eventTimeStampTable(7, referenceIdx+3);

SARSECR = zeros(7, nrOfSarsaTuples);
SARSECR(1,:) = states(1:nrOfSarsaTuples);
SARSECR(2,:) = action(1:nrOfSarsaTuples);
SARSECR(3,:) = reward(1:nrOfSarsaTuples);
SARSECR(4,:) = nextState(1:nrOfSarsaTuples);
SARSECR(5,:) = episode(1:nrOfSarsaTuples);
SARSECR(6,:) = isCatchTrial(1:nrOfSarsaTuples);
SARSECR(7,:) = reaction_time(1:nrOfSarsaTuples);
end

