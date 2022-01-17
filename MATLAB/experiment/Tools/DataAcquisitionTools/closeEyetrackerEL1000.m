function closeEyetrackerEL1000(edffilename)
% CLOSEEYETRACKEREL1000 Closes an Eyelink 1000 device
% and retreives the EDF file
%
% Based on CloseEL1000:
% Based on the Experiment Scripts by
% L. Loued-Khenissi, EPFL, June 2014
% Simplified and refactored into a function by
% Adrien Pfeuffer, Uni Marburg, July 2014

if (nargin<1)
    edffilename = 'TESTFILE';
end
% if (nargin<2)
%     logfile = 1; % stdout, i.e. the Matlab Console
% end

fprintf('Stop recording on Eyelink...\n');
% --- Commented. We have already done that in main function
%Eyelink('StopRecording');
% ---
Eyelink('CloseFile');

try
    fprintf('Trying to receiving data file ''%s''\n', edffilename);
    status=Eyelink('ReceiveFile');
    if status > 0
        fprintf('ReceiveFile status %d\n', status);
    end
    if 2==exist(edffilename, 'file')
        fprintf('Data file ''%s'' saved to ''%s''\n', edffilename, pwd);
    end
catch err
    fprintf('ERROR: Problem receiving data file ''%s''\nTry copying it manually.\n', edffilename);
    %     if (logfile ~= 1) % The user should see this immediately, not when checking the logs after overwriting the file!
    %         fprintf('ERROR: Problem receiving data file ''%s''\nTry copying it manually.\n', edffilename);
    %     end
    err;
end
fprintf('Shutting down Eyelink...\n');
Eyelink('Shutdown');

end

