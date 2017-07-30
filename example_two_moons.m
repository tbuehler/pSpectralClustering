function example_two_moons()
% Example for p-spectral clustering on high-dimensional noisy two
% moons dataset
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
    
    [X,y] = generateTwoMoons(2000,100,[0.5 0.5],0.02);
    W = buildWeightsKNN(X',false,true,10,4);

    [clusters, cuts, cheegers, vmin] = pSpectralClustering(W,1.1,true,2);

    plotClusters(X, clusters);

end


function [x,y] = generateTwoMoons(num,dim,ClassProb,variance)

    assert(length(ClassProb)==2, 'length of class probabilities should be 2');
    assert(sum(ClassProb)==1.0, 'class probabilities should sum up to 1');

    pos = rand(num,1); 
    num_pos = sum(pos<ClassProb(1));
    
    fprintf('Number of points in Class 1: %d. class 2: %d.\n', num_pos, num - num_pos);
    fprintf('Dim: %d. Noise variance: %.2f.\n', dim, variance);

    % generate two moons problem
    x = zeros(dim,num);
    y = zeros(num,1);
    
    phi = rand(num,1).*pi;
    for i=1:num_pos
        x(1,i) = cos(phi(i));
        x(2,i) = sin(phi(i));
        y(i,1) = 1;
    end
    for i=num_pos+1:num
        x(1,i) = 1+cos(phi(i));
        x(2,i) = -sin(phi(i))+0.5;
        y(i,1) = 2;
    end

    % add Gaussian noise in all dim dimensions
    x = x + sqrt(variance)*randn(dim,num);
end


% build kNN graph
function W=buildWeightsKNN(points,mutual_knn,adaptive,numKNN,alpha)

    num = size(points,1);
    XX = sum(points.*points,2);        
    dist_squared = XX(:,ones(1,num));
    dist_squared = dist_squared + dist_squared' - 2*points*points';
    
    [dist_sorted,ix] = sort(dist_squared,1);
    KNN     = ix(2:numKNN+1,:);
    KNNDist = dist_sorted(2:numKNN+1,:);

    % locally adaptive scaling parameter
    if (~adaptive)
        gamma_squared=mean(mean(KNNDist))*ones(1,num);
    else
        gamma_squared=KNNDist(numKNN,:);%mean(KNNDist);
    end
    
    % get kNN weight matrix
    W = sparse(num,num);
    for i=1:num
        W(KNN(:,i),i)=exp(-alpha/(gamma_squared(i))*KNNDist(:,i));
    end

    % W is not symmetric yet, now we symmetrize W 
    if (mutual_knn)
        W=(W+W')-abs(W-W'); W=0.5*W; 
    else
        W=(W+W')+abs(W-W'); W=0.5*W;
    end 

    W=W-spdiags(diag(W),0,num,num);
end


function plotClusters(points,clusters)

    labels = unique(clusters);
    colors = [1 0 0;0 0 1;0 1 0];
    indexClusters = cell(size(labels,1),1);
    
    figure, hold on
    for j=1:size(labels,1)
        indexClusters{j} = (clusters==labels(j));
        plot(points(1,indexClusters{j}),points(2,indexClusters{j}),'.','Color',colors(j,:),'MarkerSize',20);
    end
    hold off, axis off
end

