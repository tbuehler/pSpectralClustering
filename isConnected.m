function [connected,components]=isConnected(W)
% Checks whether a graph is connected.
%
% Usage: [connected,components]=isConnected(W)
%
% (C)2009 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University
% http://www.ml.uni-saarland.de
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

	A = W>0; % adjacency matrix

	alreadyseen = zeros(size(W,1),1);

	currentCandidates=1;

	while ~isempty(currentCandidates)
		candidates= (sum(A(:,currentCandidates),2)>0);
		alreadyseen(currentCandidates)=1;
		currentCandidates=find(candidates-alreadyseen>0);
	end

	connected = sum(alreadyseen)==size(W,2);
    
    components=alreadyseen;
    
end
