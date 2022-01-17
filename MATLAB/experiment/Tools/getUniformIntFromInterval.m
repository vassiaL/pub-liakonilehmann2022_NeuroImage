function randomNumber = getUniformIntFromInterval(lowerBound, upperBound)
        %draws a random number from a uniform distribution in [lowerBound, upperBound]
        intervalLength = upperBound - lowerBound + 1;
        randomNumber = randi(intervalLength);
        randomNumber = randomNumber+lowerBound - 1;
end
