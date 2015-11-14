function gradient = Gradient(u,W,p,normalized,deg,functional)
% Computes the gradient of the Functional F
%
% Usage: gradient = Gradient(u,W,p,normalized,deg,functional)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    W3u = getSparseDerivativeMatrix(u,W);
    W4u=computeAbsPower(W3u,p-1);
    W4ub=W4u.*sign(W3u);
    W5u= sparse(W.*W4ub);
    W5= W5u-W5u';
    
    left=sum(W5)';

    u2=abs(u).^(p-1).*sign(u);
    
    if (normalized)
        denom=pNormPowDeg(u,p,deg);
        u2=u2.*deg';
    else
        denom = pNormPow(u,p);
    end
    
    gradient= p/denom * (left-functional*u2);
    
end