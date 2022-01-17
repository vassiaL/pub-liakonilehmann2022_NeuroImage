function drawFixationCross(w, colorOfCross)
% Draws white discreet fixation cross on the screen.


crossRadius = 10;
crossLineWidth = 2;

% Find the coordinates of the center of the screen
[width, height] = Screen('WindowSize', w);
centerX = width/2;
centerY = height/2;

% jitterProbability = 0.2;
% jitterRadius = 1.5;
% if rand <= jitterProbability
%     xJitter = randi(jitterRadius*2,1,1)-jitterRadius;
%     yJitter = randi(jitterRadius*2,1,1)-jitterRadius;
%     centerX = centerX+xJitter;
%     centerY = centerY + yJitter;
% end

Screen('DrawLine', w, colorOfCross, centerX - crossRadius, centerY, centerX + crossRadius, centerY, crossLineWidth);
Screen('DrawLine', w, colorOfCross, centerX, centerY - crossRadius, centerX, centerY + crossRadius, crossLineWidth);
end