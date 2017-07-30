function [vmin,fmin,umin,normGrad,clust_iter,funct_iter] = computePEigenvector(W,p,normalized,stop_early,vmin,p_old)
% Computes the second eigenvalue/eigenvector of the graph p-Laplacian.
%
% Usage:	[vmin,fmin,umin,normGrad] = computePEigenvector(W,p,normalized,vmin,p_old)
% 			[vmin,fmin,umin,normGrad] = computePEigenvector(W,p,normalized,vmin)
% 			[vmin,fmin,umin,normGrad] = computePEigenvector(W,p,normalized)
%
% W             - Sparse weight matrix. Has to be symmetric.
% p             - Has to be in the interval ]1,2].
% normalized    - true for normalized graph p-Laplacian, false for 
%                 unnormalized graph p-Laplacian
% stop_early    - Optional: stops eigenvector computation if tresholded vector does not change anymore (default:false)
% vmin          - Optional initialization (default: 2nd eigenvector of 
%                 standard graph Laplacian)
% p_old         - Optional start value for reduction of p (default:2)
%
% vmin          - 2nd eigenvector of normalied/unnormalized graph p-Laplacian.
% fmin          - 2nd eigenvalue of normalied/unnormalized graph p-Laplacian.
% umin          - Minimizer of the functional F_2.
% normGrad      - Norm of the gradient.
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

     % Check inputs
    if ~isempty(find(W~=W',1))
        error('Weight matrix is not symmetric.');
    elseif ~issparse(W)
        error('Weight matrix is not sparse.');
    elseif ~isempty(find(diag(W)~=0,1))
        error('Graph contains self loops.');
    elseif ~isConnected(W) 
        error('Graph is not connected.');
    elseif p<=1 || p>2
        error ('p has to be in the interval ]1,2].');
    elseif nargin==6 && p_old<p
        error ('Old value of p is smaller than new value.');
    end

    
    % max number of iterations
    itmax=1000000;
    
    % accuracy
    epsilon=1E-10;
    
    % reduction factor
    factor=0.9;

    if (nargin<6)
        p_old=2;
    end;
    if (nargin<5)
        if (p<2)
            disp('...Initialising with eigenvector of standard Laplacian (p=2).');
        else
            disp('...Computing eigenvector of standard Laplacian (p=2).');
        end        
        [vmin,fmin] = SecondEigenvector(W,normalized);
        umin=vmin;
        normGrad=0;
        clust_iter=0;
        funct_iter=0;
    end
    if (nargin<4)
        stop_early=false;
    end;
    
    if (p<2)
        p_new=p_old*factor;

        while (p_new>p)

            if (p_new*factor<p)
                p_new=p_old*sqrt(p/p_old);
            end
            fprintf('...Decreasing p. Current value is p=%.3f.\n',p_new);
          
            vmin=minimizeFunctional(W,p_new,itmax,vmin,epsilon,normalized,true,stop_early);
      
            p_old=p_new;
            p_new=p_old*factor;

        end

         fprintf('...Decreasing p. Current value is p=%.3f.\n',p);
      
         [vmin,fmin,umin,normGrad,clust_iter,funct_iter]=minimizeFunctional(W,p,itmax,vmin,epsilon,normalized,false,stop_early);
   
    end
end

