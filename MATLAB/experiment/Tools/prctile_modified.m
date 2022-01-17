function y = prctile_modified(x,p)
%% We reimplemented prctile, because the stats-toolbox was not available on one of the CHUV computers.
% This code implement the NEAREST RANK METHOD which is NOT the same as
% implemented by matlab

nrOfValues = length(x);
if nrOfValues ==0
   error('data vector x is empty'); 
end
if (nrOfValues == 1)
    y= x(1);
    return
end


if (p<=0)
    y = min(x);
    return
end

if (p>=100)
    y = max(x);
    return
end

xSorted = sort(x);

n = ceil(p/100*nrOfValues);
y = xSorted(n);




