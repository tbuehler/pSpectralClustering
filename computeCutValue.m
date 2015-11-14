function [cutpart1,cutpart2] = computeCutValue(clusters,W,normalized)
% Computes the components in the Ratio/Normalized Cut and Ratio/Normalized Cheeger Cut expression. 
%
% Usage: [cutpart1,cutpart2] = computeCutValue(clusters,W,normalized)
%
% One then has Ratio/Normalized Cut = cutpart1 + cutpart2
% and Ratio/Normalized Cheeger Cut = max(cutpart1,cutpart2)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

    W2 = W(clusters==1,clusters~=1);
    cut=full(sum(sum(W2)));

    if (~normalized)
        sizeA = sum(clusters==1);
        sizeB = size(clusters,1)-sizeA;
    
        cutpart1=cut/sizeA;
        cutpart2=cut/sizeB;
    else
        degA = sum(W(:,clusters==1));
        volA = sum(degA);
    
        degB = sum(W(:,clusters~=1));
        volB = sum(degB);    
    
        cutpart1=cut/volA;
        cutpart2=cut/volB;
    end
    
end