function [triggerTime, triggerCode] = waitForScannerTrigger()

% WAITFORSCANNERTRIGGER
% function to synchronize the task with the scanner
% Borrowed from Jonas Kaplan Matlab Course 
% Employs Psychtoolbox functions (KbCheck etc).

%Key code for Trigger (5)
triggerCode = KbName('5%');
keyIsDown = 0;

%Make sure no keys are disabled
DisableKeysForKbCheck([]);

%Now wait for trigger
while 1
        [ keyIsDown, secs, keyCode ] =KbCheck (-1);
        if keyIsDown 
            if find(keyCode)==triggerCode
                break;
            end
        end
end

%Record trigger time for future reference
triggerTime = GetSecs;
fprintf('Trigger detected \n' );

DisableKeysForKbCheck(triggerCode);



end






