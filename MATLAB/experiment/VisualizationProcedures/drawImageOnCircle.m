function [imageBoxCoords]= drawImageOnCircle(w, texture,currentStatePositionId, totalNrOfStates, imageSize)
% Draws the visual cue on the screen. The position is chosen such that all
% the states lie on a circle. The same cue is shown at the same position
% during the whole experiment
[width, height] = Screen('WindowSize', w);

%icon box
if ~ exist('imageSize','var')
    edgeLength = min(width/6, height/6);
    edgeLength = min(edgeLength, 200);
    edgeLength = max(edgeLength, 50);
    sampleRect = [0 0 edgeLength edgeLength];
else
    sampleRect = [0 0 imageSize(2) imageSize(1)];
end

polarPhi = 2*pi/totalNrOfStates *(currentStatePositionId-1);

posX = (width/2) + (cos(polarPhi)*width*0.3); 
posY = (height/2) + (sin(polarPhi)*height*0.3);
imageBoxCoords = CenterRectOnPoint(sampleRect, posX, posY);

Screen('DrawTexture',w, texture,...
    [], imageBoxCoords');
%DrawFormattedText(w, sprintf('%d',currentStatePositionId), 'center', 'center');


end