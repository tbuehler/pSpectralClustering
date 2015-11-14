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

	W3u=getSparseDerivativeMatrix(u,sparse(W));
    W4u=computeAbsPower(W3u,p-2);
    W5u= sparse(W.*W4u);
    W5= W5u+W5u';
   
    D=diag(sum(W5));
   
    hessian = p*(p-1)/denom * (D-W5);

end