function [v,cest] = minimizeVariance(u,p,normalized,deg,cest,cmin,cmax)
% Computes the minimizer of the variance of u:
% c = argmin_k norm(u-k*m,p)^p
% v = u-c*m
%
% Usage: [v,c] = minimizeVariance(u,p,normalized,deg,cest,cmin,cmax)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    % Parameters for stepsize selection
    sigma=0.01;
    beta=0.3;
    eps_descent=1E-10;
    
    % Initialization
    i=0;
    flipcount=0;
    var_old=0;

    % Update value of v
    v=u-cest;
    v_abs=abs(v);
    
    % Compute gradient
    if (normalized)
        grad= p*deg*(v_abs.^(p-1).*sign(v))*(-1);
    else
        grad=-p*sum(v_abs.^(p-1).*sign(v));
    end
    gradnorm=abs(grad);
   
    % Main loop
    while (gradnorm>eps_descent && i<=100)
        
        % Compute Hessian
        if (normalized)
            hessian= (deg*v_abs.^(p-2))*(p-1)*p;
        else
            hessian=sum(v_abs.^(p-2))*(p-1)*p;
        end
        
        if (hessian==Inf)
            if (normalized)
                hessian= (deg(v_abs>0)*v_abs(v_abs>0).^(p-2))*(p-1)*p;
            else
                hessian=sum(v_abs(v_abs>0).^(p-2))*(p-1)*p;
            end
        end
        descent= -grad/hessian;

        
       if (i<2)
            % Full Newton step
            cest_new=cest+descent;
            
            if (cest_new>cmax)
                cest_new=cmax;
            elseif (cest_new<cmin)
                cest_new=cmin;
            end
       else
            % Newton step with reduced stepsize
            cest_new=cest+descent;
            if (cest_new>cmax)
                stepsize_init= (cmax-cest)/descent;
            elseif (cest_new<cmin)
                stepsize_init= (cmin-cest)/descent;
            else
                stepsize_init=1;
            end
           
            if (i==2)
                if (normalized)
                    var_old = pNormPowDeg(v,p,deg);
                else
                    var_old = pNormPow(v,p);
                end
            end
          
           [cest_new,~,var_old] = makestep(cest,descent,grad,u,var_old,p,sigma,beta,normalized,deg,stepsize_init);
        end
        
        % If the gradient keeps flipping signs, just take the mean of the last two iterates
        if (flipcount==2 || i==100)
            cest=(cest+cest_new)/2;
            flipcount=0;
        else
            cest=cest_new;
        end
             
        % Update value of v
        v=u-cest;
        v_abs=abs(v);
        
        % Compute gradient
        if (normalized)
            grad_new= p*deg*(v_abs.^(p-1).*sign(v))*(-1);
        else
            grad_new=-p*sum(v_abs.^(p-1).*sign(v));
        end
        
        % Update flipcount
        if sign(grad_new)~=sign(grad)
            flipcount=flipcount+1;
        else
            flipcount=0;
        end
                
        grad=grad_new;
        gradnorm=abs(grad);
        
        i=i+1;
      
    end

end

% Makes a step using Armijo stepsize selection
function [c_new,stepsize,var_new] = makestep(c_old,descent,graduval,v_old,var_old,p,sigma,beta,normalized,deg,stepsize_init)

    epsilon=1E-14;
    
    iter=1;
    stepsize=stepsize_init;

    c_new=c_old+stepsize*descent;
    v_new=v_old-c_new;
    
    if (normalized)
        var_new = pNormPowDeg(v_new,p,deg);
    else
        var_new = pNormPow(v_new,p);
    end
    
    leftside=(var_new - var_old)/var_new;
    rightside=sigma*stepsize*graduval*descent;
   
    while(leftside>rightside+epsilon && stepsize>=epsilon)
        iter=iter+1;
        stepsize=stepsize*beta;

        c_new=c_old+stepsize*descent;
    
        v_new=v_old-c_new;
        if (normalized)
            var_new = pNormPowDeg(v_new,p,deg);
        else
            var_new = pNormPow(v_new,p);
        end
    
        leftside=(var_new - var_old)/var_new;
        rightside=rightside*beta;
    end
end

