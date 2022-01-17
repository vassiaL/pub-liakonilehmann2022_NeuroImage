function [setUp, filename, pathname] = expSetUp()
% Gets user inputs for Subject ID , Run ID and Subject's additional info.
% Opens a dialog box for saving files.
% If file already exists, a warning dialog box opens stating that the file
% already exists and asks if user wants to replace it.
% Returns the user inputs, the path and the filename where results will be saved.

defaultValues = {'01', '01', '-'};

%% Get user inputs
% Prompt line
prompt = 'Press ANY KEY and ENTER for set up. Press ENTER to skip and use defaults. ';
answer = input(prompt, 's');

if ~isempty(answer)
    [setUp, filename, pathname] = getSetUpAnswer(defaultValues);
    
else % Proceed with defaults
    disp('Proceeding with default values. ');
    
    setUp.subjId = defaultValues{2};
    setUp.runId = defaultValues{2};
    setUp.info = defaultValues{3};
    
    pathname = './' ;
    filename = strcat('SubjId',setUp.subjId,'_RunId',setUp.runId,'.mat');
    
    % Check if file exists
    checkIfExists = which(fullfile(pathname,filename));
    if ~isempty(checkIfExists)
        disp('Warning: File exists');
        prompt = 'Press ANY KEY and ENTER to change name. Press ENTER to proceed anyway. ';
        answer = input(prompt, 's');
        
        if ~isempty(answer)
            [setUp, filename, pathname] = getSetUpAnswer;
        else
            disp(strcat('Results will be saved in: ',fullfile(pathname,filename)));
        end
    else
        disp(strcat('Results will be saved in: ',fullfile(pathname,filename)));
    end
end


end



function [setUp, filename, pathname] = getSetUpAnswer(defaultValues)

%% Get Subject ID
prompt = 'Enter the Subject ID: ';
answer = input(prompt, 's');

if isempty(answer);
    disp('No subject ID provided. ');
    disp('Proceeding with default value. ');
    setUp.subjId = defaultValues{1};
else
    setUp.subjId = answer;
end

%% Get Run ID
prompt = 'Enter the Run ID: ';
answer = input(prompt, 's');

if isempty(answer);
    disp('No Run ID provided. ');
    disp('Proceeding with default value. ');
    setUp.runId = defaultValues{2};
else
    setUp.runId = answer;
end

%% Get additional info
prompt = 'Enter additional info: ';
answer = input(prompt, 's');

if isempty(answer);
    disp('No additional info provided. ');
    disp('Proceeding with default value. ');
    setUp.info = defaultValues{3};
else
    setUp.info = answer;
end


%% Open standard dialog box for saving files
[filename, pathname] = uiputfile(strcat('SubjId',setUp.subjId,'_RunId',setUp.runId,'.mat'),'Save results');

% Check if cancel was selected or diplay full path name
if isequal(filename,0) || isequal(pathname,0)
    disp('Cancel was selected.');
else
    disp(strcat('Results will be saved in: ',fullfile(pathname,filename)));
end
end