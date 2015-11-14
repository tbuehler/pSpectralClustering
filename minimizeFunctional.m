function [vmin,fmin,umin,normgrad]=minimizeFunctional(W,p,itmax,u,epsilon,normalized,intermediate)
% Computes the minimum of the variational characterization of the second eigenvector of the 
% graph p-Laplacian via a combination of steepest descent and Newton descent.
% 
% Usage: [vmin,fmin,umin,normgrad]=minimizeFunctional(W,p,itmax,u,epsilon,normalized,intermediate)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    eps_descent=1E-6;
    
    num=length(u);

    if(intermediate)
        maxNoClusterChangeIterations=12/(p*p);
    else
        maxNoClusterChangeIterations=30;
    end
    
    maxGradient=10;
    maxNoFunctionalChangeIterations=10;
    NoFunctionalChangeIterations=0;
    NoClusterChangeIterations=0;
    
	threshold_type=-1;
    criterion=2;

    u=u/norm(u);
    deg=full(sum(W));% row vector

    i=0;
    FLAG=1;

    c=0.5*(min(u)+max(u));
    
    v=minimizeVariance(u,p,normalized,deg,c,min(u),max(u));
     
    oldClusters =  createClusters(v,W,normalized,threshold_type,criterion);
   
    currentF=Functional(v,W,p,normalized,deg);
    grad=Gradient(v,W,p,normalized,deg,currentF);
    gradNorm=norm(grad)/sqrt(size(v,1));
    
  
    gradientCount=0;
    tryNewton=true;

    
    descent=zeros(size(u,1),1);
    armijo=true;
    
    temp_tic=tic;
     
    while(FLAG && i<itmax)
        i=i+1;
        
        if (tryNewton)
          
            Hess=Hessian(v,W,p,normalized,deg);
           
            scaling=max(max(abs(Hess)));          
            Hess=Hess/scaling;
        
            lambda=1E-10;
            Hess3= Hess+ lambda*speye(num);
            
            warning off all;
            [descent3,flag] = minres(Hess3,-grad,1E-6,1000,[],[],descent*scaling);
            warning on all;
            
            descent=descent3/scaling;
          
            funct_change=descent'*grad;
            
            if (currentF+funct_change<=0 || isnan(funct_change)) 
                descent=-grad;
       
                Newton=false;
                gradientCount=gradientCount+1;
                         
                [u,v,newF] = makeStep(v,currentF,grad,descent,W,p,epsilon,normalized,deg,Newton,armijo);
            else
                [u_N,v_N,newF_N] = makeStep(v,currentF,grad,descent,W,p,epsilon,normalized,deg,true,armijo);
                
                descent=-grad;
                [u_G,v_G,newF_G] = makeStep(v,currentF,grad,descent,W,p,epsilon,normalized,deg,false,armijo);
               
                diff_F=(newF_N-newF_G)/newF_N;
                if(1-newF_N/currentF<=-epsilon || diff_F>epsilon*100)
                    u=u_G;
                    v=v_G;
                    newF=newF_G;
                       
                    Newton=false;
                    gradientCount=gradientCount+1;
                else
                    u=u_N;
                    v=v_N;
                    newF=newF_N;
                               
                    Newton=true;
                end
            end
        else
            descent = -grad;
            
            Newton=false;
            gradientCount=gradientCount+1;
                      
            [u,v,newF] = makeStep(v,currentF,grad,descent,W,p,epsilon,normalized,deg,Newton,armijo);
        end

        
        % Check if functional has changed
        functChange=1-newF/currentF;
        if (abs(functChange)< epsilon*100);
            NoFunctionalChangeIterations=NoFunctionalChangeIterations+1; 
        else 
            NoFunctionalChangeIterations=0;
        end
        
     
        % Check if clusters have changed 
        allClusters =  createClusters(v,W,normalized,threshold_type,criterion);
   
        changeFlag=sum(allClusters~=oldClusters);
        oldClusters=allClusters;
        
        if (changeFlag==0)
            if (Newton || gradientCount==maxGradient)
                NoClusterChangeIterations=NoClusterChangeIterations+1;
            end
        else
            NoClusterChangeIterations=0;
        end
      

        % Compute gradient
        grad=Gradient(v,W,p,normalized,deg,newF);
        gradNorm=norm(grad)/sqrt(size(v,1)); 
            

        % Decide whether to do a Newton step or gradient step
        tryNewton=false;
        if(gradientCount==maxGradient) % if we have done a number of gradient steps
                tryNewton=true; 
                gradientCount=0;
        elseif(Newton)   %if last step was a (successful) Newton step
            tryNewton=true;
        end
        
        
        % Decide which step size selection method to use
        if (NoFunctionalChangeIterations>1 || NoClusterChangeIterations>25 || gradNorm< eps_descent*5)
            armijo=false;
        else
            armijo=true;
        end
 
        
        % Check if converged
        if (gradNorm <eps_descent)
            FLAG=0; 
        end;
        if (NoFunctionalChangeIterations==maxNoFunctionalChangeIterations)
            FLAG=0;
        end
        if (NoClusterChangeIterations>=maxNoClusterChangeIterations)
            FLAG=0;
        end
        
        
        % Update iterate
        currentF=newF;
        
        % Make a status message every 2 minutes
        temp_toc=toc(temp_tic);
        if temp_toc>120
            temp_tic=tic;
            fprintf('   Current inner iteration: %d. Functional value: %.8g. Gradient norm: %.8g.\n',i,currentF,gradNorm);
        end
    end

 
    % the output 
    normgrad=gradNorm;
    umin = u;                                 % optimal point
    fmin = currentF;                          % optimal objective value
    vmin=v;

end
  
%% Makes a step
function [newU,newV,newF,stepsize]=makeStep(v,currentF,graduval,descent,W,p,epsilon,normalized,deg,Newton,Armijo)  % returns the stepsize
 
    if(Armijo)
        if(Newton)
            sigma=1E-4;
            beta=0.5;
            [newU,newV,newF,stepsize]=makeStepArmijo(v,currentF,graduval,descent,sigma,beta,W,p,epsilon,normalized,deg); 
        else
            sigma=1E-4;
            beta=0.1;
            [newU,newV,newF,stepsize]=makeStepArmijoGradient(v,currentF,graduval,descent,sigma,beta,W,p,epsilon,normalized,deg);
        end
    else
        [newU,newV,newF,stepsize]=makeStepGoldenSectionLineSearch(v,descent,W,p,epsilon,normalized,deg); 
    end
    
end

%% makes a step by selecting the step size via a golden section line search
function [newU,newV,newF,stepsize]=makeStepGoldenSectionLineSearch(v,descent,W,p,epsilon,normalized,deg)
       
        a = epsilon;
        b = 1;
    
        stepsize=a;
        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);    
        newA=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
        aval=Functional(newA,W,p,normalized,deg);
    
        stepsize=b;
        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);       
        newB=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
        bval=Functional(newB,W,p,normalized,deg);
         

        tau=(3-sqrt(5))/2;
    
        mu = a + tau*(b-a);
        nu = b - tau*(b-a);
    
        stepsize=mu;        
        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);       
        newMu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
        muval=Functional(newMu,W,p,normalized,deg);
    
        stepsize=nu;
        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);  
        newNu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
        nuval=Functional(newNu,W,p,normalized,deg);
 
        while (b-a>epsilon)
            if(muval>aval && nuval>bval)
                if (aval<bval)
                    bval=muval;
                    b=mu;
                else
                    aval=nuval;
                    a=nu;
                end
                mu = a + tau*(b-a);
                nu = b - tau*(b-a);
    
                stepsize=mu;        
                newU=computeNewU(v,stepsize,descent);
                [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);  
                newMu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
                muval=Functional(newMu,W,p,normalized,deg);
    
                stepsize=nu;
                newU=computeNewU(v,stepsize,descent);
                [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);  
                newNu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
                nuval=Functional(newNu,W,p,normalized,deg);
            else
                [a,aval,b,bval,mu,muval,nu,nuval] = updateBounds(a,aval,b,bval,mu,muval,nu,nuval,v,W,p,descent,tau,normalized,deg);
            end
        end

        [newF,ind]=min([aval,muval,nuval,bval]);
        candidates=[a,mu, nu,b];
        stepsize=candidates(1,ind(1));
                
        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);     
        newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     

end

%% makes a step with Armijo stepsize selection
function [newU,newV,newF,stepsize]=makeStepArmijo(v,currentF,graduval,descent,sigma,beta,W,p,epsilon,normalized,deg)  % returns the stepsize

	stepsize=1;

    newU=computeNewU(v,stepsize,descent);
    [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);  
    newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
        
    newF = Functional(newV,W,p,normalized,deg);
    
    leftside=newF-currentF;
    rightside=sigma*stepsize*graduval'*descent;

   
    while(leftside>rightside && stepsize>=epsilon)
        stepsize=stepsize*beta;

        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
        newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
                  
        newF = Functional(newV,W,p,normalized,deg);
       
        leftside=newF-currentF;
        rightside=rightside*beta;
    end
    
end




%% makes a step with Armijo stepsize selection
function [newU,newV,newF,stepsize]=makeStepArmijoGradient(v,currentF,graduval,descent,sigma,beta,W,p,epsilon,normalized,deg)  % returns the stepsize
iter=1;
	stepsize=1;

    newU=computeNewU(v,stepsize,descent);
    [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);  
    newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
        
    newF = Functional(newV,W,p,normalized,deg);
    
    leftside=newF-currentF;
    rightside=sigma*stepsize*graduval'*descent;
   
    while(leftside>rightside && stepsize>=epsilon)
        iter=iter+1;
        stepsize=stepsize*beta;

        newU=computeNewU(v,stepsize,descent);
        [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
        newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);
                  
        newF = Functional(newV,W,p,normalized,deg);
       
        leftside=newF-currentF;
        rightside=rightside*beta;

    end
    
end


%% Computes the new iterate
function newU=computeNewU(v,stepsize,descent)
        
    newU=v+stepsize*descent;

end

%% Calculates estimates for the variance minimizer
function [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize)
        
    estimate = stepsize*median(descent);
	leftestimate = stepsize*min(descent);
	rightestimate = stepsize*max(descent);

end


%% Updates the bounds around the best solution    
function [a,aval,b,bval,mu,muval,nu,nuval]=updateBounds(a,aval,b,bval,mu,muval,nu,nuval,v,W,p,descent,tau,normalized,deg)

    if (muval<nuval)
       if (aval <= muval)
            b=mu;
            bval=muval;
      
            [mu,nu,muval,nuval]=computeMuNu(a,b,v,W,p,descent,tau,normalized,deg);
        else
            b=nu;
            bval=nuval;
      
            nu = mu;
            nuval=muval;
            
            mu = a + tau*(b-a);
            stepsize=mu;
            newU=computeNewU(v,stepsize,descent);
            [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
            newMu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
            muval=Functional(newMu,W,p,normalized,deg);
        end
    elseif (muval>nuval)
        if (nuval>=bval)
            a=nu;
            aval=nuval;
            
           [mu,nu,muval,nuval]=computeMuNu(a,b,v,W,p,descent,tau,normalized,deg);
        else
            a=mu;
            aval=muval;
            
            mu = nu;
            muval=nuval;
            
            nu = b- tau*(b-a);
            stepsize=nu;
            newU=computeNewU(v,stepsize,descent);
            [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
            newNu=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
            nuval=Functional(newNu,W,p,normalized,deg);
        end  
    else
        a=mu;
        aval=muval;
        
        b=nu;
        bval=nuval;
        
        [mu,nu,muval,nuval]=computeMuNu(a,b,v,W,p,descent,tau,normalized,deg);
    end
           
end

%% Computes new test points for interval boundary
function [mu,nu,muval,nuval]=computeMuNu(a,b,v,W,p,descent,tau,normalized,deg)

    step= tau*(b-a);
            
    mu = a + step;
    stepsize=mu;
    newU=computeNewU(v,stepsize,descent);
    [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
    newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
    muval=Functional(newV,W,p,normalized,deg);
    
    nu = b- step;
    stepsize=nu;
    newU=computeNewU(v,stepsize,descent);
    [leftestimate, estimate, rightestimate]=calculateEstimates(descent,stepsize);        
    newV=minimizeVariance(newU,p,normalized,deg,estimate,leftestimate,rightestimate);     
    nuval=Functional(newV,W,p,normalized,deg);
end
