function hessian = Hessian(u,W,p,normalized,deg)
% Computes the hessian of the Functional F
%
% hessian = Hessian(u,W,p,normalized,deg)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    if (normalized)
        denom=pNormPowDeg(u,p,deg);
    else
        denom=pNormPow(u,p);
    end

    W3=getSparseDerivativeMatrix(u,sparse(W));
    W3=computeAbsPower(W3,p-2);
    W4= sparse(W.*W3);
    W4= W4+W4';
   
    D=diag(sum(W4));
   
    hessian = p*(p-1)/denom * (D-W4);

end
