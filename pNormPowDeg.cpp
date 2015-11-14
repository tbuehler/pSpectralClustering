// (C)2009 Thomas Buehler and Matthias Hein
// Machine Learning Group, Saarland University
// http://www.ml.uni-saarland.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
             
  mwIndex i,m,n,size;
  double * p;
  double * vector;
  double * deg;
  double * sumPtr;
  double sum;

  // Test number of parameters.
  if (nrhs != 3 || nlhs != 1) {
    mexWarnMsgTxt("Usage: n = pNorm(u,p)\n");
    return;
  }
  
  // Parse parameters
  m = mxGetM(prhs[0]);
  n = mxGetN(prhs[0]);
  vector = mxGetPr(prhs[0]);
  
  p = mxGetPr(prhs[1]);
  deg= mxGetPr(prhs[2]);
 
  plhs[0] = mxCreateDoubleScalar(0);
  sumPtr = mxGetPr(plhs[0]);

  size=m*n;

  sum=0;
  
  // computed weighted p-norm to power p
  for (i=0;i<size;i++)
  {   
      sum += deg[i]*pow(fabs(vector[i]),*p);
  }
  
  
  sumPtr[0] = sum;
}