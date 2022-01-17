function imageNameCellArray = getRandomImageNames(nrOfSelectedImages, totalNrOfImages)
% Pick nrOfSelectedImages images randomly from a pool of totalNrOfImages
% available images

imageNameCellArray = cell(1,nrOfSelectedImages);
iPermuation =  randperm(totalNrOfImages);
for i = 1:nrOfSelectedImages
    imageNameCellArray{i} = strcat('I', num2str(iPermuation(i)), '.jpg');
end

end

