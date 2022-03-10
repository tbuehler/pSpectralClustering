function [clusters,cuts,cheegers,vmin,fmin,normGrad,clust_iter,funct_iter] = pSpectralClustering(W,p,normalized,k,stop_early)
% Computes a multipartitioning of the data given by a similarity matrix W 
% by recursively computing bipartitions using the second eigenvector of 
% the graph p-Laplacian.
%
% Usage:	[clusters,cuts,cheegers,vmin,fmin,normGrad] = computeClustering(W,p,normalized,k)
%
% W             - Sparse weight matrix. Has to be symmetric.
% p             - Has to be in the interval ]1,2].
% normalized    - true for Ncut/NCC, false for Rcut/RCC
% k             - number of clusters
% stop_early    - Optional: stops eigenvector computation if tresholded vector does not change anymore (default:true).
%                 set to false to compute eigenvector more accurately
%
% clusters      - mx(k-1) matrix containing in each column the computed 
%                 clustering for each partitioning step.
% cuts          - (k-1)x1 vector containing the Ratio/Normalized Cut values 
%                 after each partitioning step.
% cheegers      - (k-1)x1 vector containing the Ratio/Normalized Cheeger 
%                 Cut values after each partitioning step.
% vmin          - Second eigenvector of the graph p-Laplacian for the 
%                 first partitioning step.
% fmin          - Second eigenvalue of the graph p-Laplacian for the first 
%                 partitioning step.
% normGrad      - Norm of the gradient after the first partitioning step.
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
        
    if (nargin<5) stop_early = true; end

    threshold_type = -1; % 0: zero, 1: median, 2: mean, -1: best
    criterion_threshold = 2; % 1: Ratio/Normalized Cut, 2: Ratio/Normalized Cheeger Cut
    criterion_multicluster = 1; % 1: Ratio/Normalized Cut, 2: Ratio/Normalized Cheeger Cut
        
    successive = true; % false for direct minimization
    
    clusters = zeros(size(W,1),k-1);
    cuts = zeros(1,k-1);
    cheegers = zeros(1,k-1);

    cutParts = zeros(1,k);
   
    if (k<2)
        error ('Number of clusters has to be at least 2.');
    elseif (k>size(W,1))
        error ('Number of clusters is larger than size of the graph.');
    end      
   
    deg = full(sum(W));
        
    fprintf('\nComputing partitioning for p=%.3f.\n',p);
    if (successive)
        [vmin,fmin,umin,normGrad,clust_iter,funct_iter] = computePEigenvector(W,p,normalized,stop_early);
    else
        start = randn(size(W,1));
        [vmin,fmin,umin,normGrad] = minimizeFunctional(W,p,1000000,start,1E-10,normalized,false);
    end
     
    disp('...Computing cluster indicator function.');
          
    [allClusters,cut,cheeger,cutPart1,cutPart2] = createClusters(vmin,W,normalized,threshold_type,criterion_threshold);
    
    allClusters = allClusters+1;
    clusters(:,1) = allClusters;
    cuts(:,1) = cut;
    cheegers(:,1) = cheeger;

    cutParts(1) = cutPart1;
    cutParts(2) = cutPart2;

    subCutParts = zeros(k,2);
    subClusters = cell(1,k);

    fprintf('Finished Clustering into 2 parts.\n');
    if (normalized)
        fprintf('Normalized Cut: %.8g   Normalized Cheeger Cut: %.8g\n\n',cut,cheeger); 
    else
        fprintf('Ratio Cut: %.8g   Ratio Cheeger Cut: %.8g\n\n',cut,cheeger); 
    end
    
    for l=3:k
        bestCut = inf;
        bestCheeger = inf;
        for m=1:l-1
            indexM = find(allClusters==m);
            if (length(indexM)==1) continue; end
               
            if (~isempty(subClusters{m}))
                allClustersInClusterM = subClusters{m};
                cutPart1 = subCutParts(m,1);
                cutPart2 = subCutParts(m,2);
                
            else    
                Wm = W(indexM,indexM);
                [connected,components]=isConnected(Wm);
                
                if (connected)
                    fprintf('Computing partitioning of subgraph %d for p=%.3f.\n',m,p);
                    if (successive)
                        vminM = computePEigenvector(Wm,p,normalized,stop_early);
                    else
                        startM = randn(size(Wm,1));
                        vminM = minimizeFunctional(Wm,p,1000000,startM,1E-10,normalized,false);
                    end
	
                    fprintf('...Computing cluster indicator function.\n\n');
                    if threshold_type>=0
                        threshold = determineThreshold(threshold_type,vminM);
                        allClustersInClusterM = vminM>threshold;
                                                
                        clusterM1 = zeros(size(allClusters,1),1);
                        clusterM1(indexM) = allClustersInClusterM;
                        cutPart1 = computeCutValue(clusterM1,W,normalized); % vmin< threshold

                        clusterM2 = zeros(size(allClusters,1),1);
                        clusterM2(indexM) = (allClustersInClusterM==0);
                        cutPart2 = computeCutValue(clusterM2,W,normalized); % vmin>threshold
                    else
                        % W has to have diagonal zero.
                        [vminM_sorted, index] = sort(vminM);
                        W_sorted = Wm(index,index);

                        % calculate cuts
                        degM = deg(indexM);
                        volumes_threshold = cumsum(degM(index));
                        triup = triu(W_sorted,1);
                        tempcuts_threshold = volumes_threshold - 2*cumsum(full(sum(triup))) ...
                                             - cumsum(full(diag(W_sorted)))';
                        tempcuts_threshold2 = (volumes_threshold(end)-volumes_threshold) ...
                                              - (sum(sum(W_sorted))-2*cumsum(full(sum(triup,2)))' - cumsum(full(diag(W_sorted)))');             
						
                        % divide by size/volume
                        if(normalized)
                            cutparts1_threshold = tempcuts_threshold(1:end-1)./volumes_threshold(1:end-1);
                            cutparts2_threshold = tempcuts_threshold2(1:end-1)./ (volumes_threshold(end)-volumes_threshold(1:end-1));
                        else
                            sizes_threshold = cumsum(ones(1,size(vminM,1)-1));
                            cutparts1_threshold = tempcuts_threshold(1:end-1)./sizes_threshold;
                            cutparts2_threshold = tempcuts_threshold2(1:end-1)./(size(vminM,1)-sizes_threshold);
                        end
						
                        %calculate total cuts
                        cuts_threshold = cutparts1_threshold + cutparts2_threshold;
                        cheegers_threshold = max(cutparts1_threshold,cutparts2_threshold);

                        if (criterion_threshold==1)
                            [cut,threshold_index] = min(cuts_threshold);
                        else
                            [cheeger,threshold_index] = min(cheegers_threshold);
                        end

                        cutPart1 = cutparts1_threshold(threshold_index);
                        cutPart2 = cutparts2_threshold(threshold_index);
					
                        allClustersInClusterM = vminM>vminM_sorted(threshold_index);
                    end
					
                    subClusters{m} = allClustersInClusterM;
                    subCutParts(m,1:2) = [cutPart1 cutPart2];
                else
                    allClustersInClusterM = components;
                    
                    clusterM1 = zeros(size(allClusters,1),1);
                    clusterM1(indexM) = allClustersInClusterM;
                    cutPart1 = computeCutValue(clusterM1,W,normalized); % vmin< threshold

                    clusterM2 = zeros(size(allClusters,1),1);
                    clusterM2(indexM) = (allClustersInClusterM==0);
                    cutPart2 = computeCutValue(clusterM2,W,normalized); % vmin>threshold
                    
                    subClusters{m} = allClustersInClusterM;
                    subCutParts(m,1:2) = [cutPart1 cutPart2];
                end
            end
            
            cut = sum(cutParts)-cutParts(m)+cutPart1+cutPart2;
            cheeger = max([cutParts((1:l-1)~=m) cutPart1 cutPart2]);
               
            if (criterion_multicluster==1 && cut<bestCut) || (criterion_multicluster==2 && cheeger<bestCheeger)
                bestCut = cut;
                bestCheeger = cheeger;
                bestCutPart1 = cutPart1;
                bestCutPart2 = cutPart2;
                bestM = m;
                clusters_new = allClusters;
                clusters_new(indexM) = (l-m)*allClustersInClusterM+clusters_new(indexM);
            end
        end
        allClusters = clusters_new;
        cuts(1,l-1) = bestCut;
        cheegers(1,l-1) = bestCheeger;
        
        clusters(:,l-1) = allClusters;
        
        if (sum(allClusters==bestM)>=2)
            subCutParts(bestM,:) = 0;
            subClusters{bestM} = [];
        else
            subCutParts(bestM,1:2) = [inf inf];
            subClusters{bestM} = 1;
        end
        
        if (sum(allClusters==l)>=2)
            subCutParts(l,:) = 0;
            subClusters{l} = [];
        else
            subCutParts(l,1:2) = [inf inf];
            subClusters{l} = 1;
        end
        
        cutParts(bestM) = bestCutPart1;
        cutParts(l) = bestCutPart2;

        fprintf('Decided to partition subgraph %d. Finished Clustering into %d parts.\n',bestM,l);
        
        if (normalized)
            fprintf('Normalized Cut: %.8g \n\n',bestCut);
        else
            fprintf('Ratio Cut: %.8g \n\n',bestCut);
        end
    end
end

