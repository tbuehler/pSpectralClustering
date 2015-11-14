function result = Functional(u,W,p,normalized,deg)
% Computes the value of the functional F introduced in 
% "Spectral Clustering based on the graph p-Laplacian".
%
% Usage: result = Functional(u,W,p,normalized,deg)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    W3=getSparseDerivativeMatrix(u,W);
    W3=computeAbsPower(W3,p);
    W4=sparse(W.*W3);
    
    enum = full(sum(sum(W4)));

    if (normalized)
        denom=pNormPowDeg(u,p,deg);
    else
        denom=pNormPow(u,p);
    end
    
    result = enum/denom;
    
end

