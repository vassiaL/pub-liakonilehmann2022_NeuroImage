function [valuesEvent, timeStampsEvent, eventIndices] = getTimeStampsOfAnEvent(eventTimeStampTable, eventRowNumbers)
% Get the non-zero values and the corresponding timeStamps of one event (eg action, state, isi onset, RPE,...)

%[resultRows, ~, ~] = getTechEnums;

timeStampsEvent = eventTimeStampTable(1,:);
eventRow = eventTimeStampTable(eventRowNumbers,:);
% eventIndices = find(eventRow~=0);
eventIndices = find(any(eventRow,1));

valuesEvent = eventRow(:, eventIndices);
timeStampsEvent = timeStampsEvent(1, eventIndices);

end
