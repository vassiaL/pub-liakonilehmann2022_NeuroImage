function el = startEyetrackerEL1000(window, edffilename, dummymode)
% STARTEYETRACKEREL1000 Initializes an Eyelink 1000 device
% and returns a device handle 
% Requires a PTB Window-Pointer and a name for the edffile
% Make sure the entered edffilename is 1 to 8 
% characters in length and only numbers or letters are allowed.
%
% Based on STARTEL1000:
% Based on the Experiment Scripts by
% L. Loued-Khenissi, EPFL, June 2014
% Simplified and refactored into a function by
% Adrien Pfeuffer, Uni Marburg, July 2014

if (nargin<2)
    edffilename = 'TESTFILE';
end
% if (nargin<3)
%     logfile = 1; % stdout, i.e. the Matlab Console
% end
if (nargin<3)
    dummymode = 0;
end

% Provide Eyelink with details about the graphics environment
% and perform some initializations. The information is returned
% in a structure that also contains useful defaults
% and control codes (e.g. tracker state bit and Eyelink key values).
fprintf('Initializing Eyelink...\n');
% --- The following is included in Leyla's script, but not in Andrien's.
% --- Commented for now. Basically ListenChar(2) makes it so characters typed
% --- don't show up in the command window. If you uncomment do not forget to use ListenChar(0);
% --- at the end of this script. 
% --- Better avoid it and if problem do calibration manually, because that
% --- might cause problems with capturing subject's responses
% Disable key output to Matlab window:
% ListenChar(2);
% ---
el=EyelinkInitDefaults(window);

% Initialization of the connection with the Eyelink Gazetracker.
% exit program if this fails.
if ~EyelinkInit(dummymode, 1)
    fprintf('ERROR: Eyelink Init aborted.\n');
    Eyelink('Shutdown');  % All is lost; exit gracefully.
    el = []; % There is no Eyelink data to return; don't confuse the user.
    return;
end

[v vs]=Eyelink('GetTrackerVersion');
fprintf('Running experiment on a ''%s'' tracker.\n', vs );

% open file to record data to
Eyelink('Openfile', edffilename);
fprintf('Eyelink data will be saved to ''%s.edf''\n', edffilename );

% Calibrate the eye tracker
EyelinkDoTrackerSetup(el);

% Drift correct right after calibration is pointless
% and might even make the calibration worse. -> removed.

% ---- Commented
% Start recording eye position
%Eyelink('StartRecording');
% Record a few samples before we actually start displaying
%WaitSecs(0.1);
% ----

% Mark zero-plot time in data file
Eyelink('Message', 'SYNCTIME');

% ---
% ListenChar(0);
% ---
end

