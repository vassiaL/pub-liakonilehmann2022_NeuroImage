% Delete some fields of the recorded data to ensure anonymity

dataFolder = pwd;

dataFolderContent = what(dataFolder);
allOrigMatFileNames = dataFolderContent.mat;

nrOfMatFiles = numel(allOrigMatFileNames);


for iFile = 1:nrOfMatFiles
    
    fileName = allOrigMatFileNames{iFile};
    
    fullFileName = fullfile(dataFolderContent.path, fileName);
    disp(fullFileName);
    load(fullFileName);

   
    taskResults.StartTime = '';
    taskResults.taskRunner = '';
    taskResults.EndTime = '';

    taskResults.TaskDefinition.pathname = '';
    taskResults.TaskDefinition.filename = '';
    taskResults.TaskDefinition.edffilename = '';
    taskResults.TaskDefinition.EnvModelImpl = '';

    taskResults.TaskDefinition.setUp.subjId = '';
    taskResults.TaskDefinition.setUp.runId = '';
    taskResults.TaskDefinition.setUp.info = '';
    
    save(fullfile(fullFileName),'taskResults');
    
end