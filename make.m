function make()

   fprintf('Compiling mexfiles ...');
	
    mex -largeArrayDims  computeAbsPower.cpp 
    mex -largeArrayDims  getSparseDerivativeMatrix.cpp 
    mex -largeArrayDims  pNormPow.cpp
    mex -largeArrayDims  pNormPowDeg.cpp 

    fprintf('\nThe mexfiles have been successfully created.\n');
 
end
