function [vmin,fmin]= SecondEigenvector(W,normalized)
% Computes the second eigenvector of the standard graph Laplacian.
%
% Usage: [vmin,fmin]= SecondEigenvector(W,normalized)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    opts.disp = 0;
    opts.tol = 1E-4;
    opts.maxit = 1000;  

    D=sparse(diag(sum(W)));
    Lp=D-W;
 
    if(normalized)
        [eigVecs,eigVals] =eigs(Lp,D,2,'smallestabs',opts);
    else
        [eigVecs,eigVals] =eigs(Lp,2,'smallestabs',opts);
    end
    
    vmin=eigVecs(:,2);
    vmin=vmin/norm(vmin);
    fmin=eigVals(2,2);
    
end
