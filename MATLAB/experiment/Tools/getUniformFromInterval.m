function randomNumber = getUniformFromInterval(lowerBound, upperBound)
        %draws a random number from a uniform distribution in [lowerBound, upperBound]
        intervalLength = upperBound - lowerBound;
        randomNumber = rand*intervalLength;
        randomNumber = randomNumber+lowerBound;
end
