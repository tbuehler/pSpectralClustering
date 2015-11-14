// (C)2009 Thomas Buehler and Matthias Hein
// Machine Learning Group, Saarland University
// http://www.ml.uni-saarland.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  
  mwSize sizeWm,sizeWn;
  mxArray * v;
  mwSize nzmax;
  mwIndex *ir, *jc, *irs, *jcs;
  double *weights,*sr;
  double * p;
  mwIndex currentEntryIndex,currentColumnIndex,numColumnEntries;
  mwIndex i;
             
  // Test number of parameters.
  if (nrhs != 2 || nlhs != 1) {
    mexWarnMsgTxt("Usage: K = computePower(W,p)");
    return;
  }
  
  // Parse parameters
  if (!mxIsSparse(prhs[0])) {
    mexWarnMsgTxt("Error: Expects sparse matrix W.");
    return;
  }
  sizeWm=mxGetM(prhs[0]);
  sizeWn=mxGetN(prhs[0]);
    
  nzmax = mxGetNzmax(prhs[0]); // number of nonzero elements of sparse matrix
  ir = mxGetIr(prhs[0]);
  jc = mxGetJc(prhs[0]);
  weights = mxGetPr(prhs[0]);
    
  p = mxGetPr(prhs[1]);
  // Allocate memory for output (sparse real matrix)
  v = mxCreateSparse(sizeWm, sizeWn, nzmax, mxREAL);
  sr = mxGetPr(v);
  irs = mxGetIr(v);
  jcs = mxGetJc(v);
  
  
  currentEntryIndex=0;
  currentColumnIndex=0;
  numColumnEntries=0;
  
  jcs[0]=0;
 
  
  // compute the power p for every entry
  for (i=0;i<nzmax;i++)
  {
      if (i<sizeWn+1) //jc has only dimension sizeWn+1
          jcs[i]=jc[i];
      irs[i]=ir[i];
      sr[i]=pow(fabs(weights[i]),*p);
  }
  
  // if we have less than sizeWn entries
  if (nzmax < sizeWn+1)
  {
      for (i=nzmax;i<sizeWn+1;i++)
          jcs[i]=jc[i];
  }
      
  
   plhs[0]=v; 
 }
