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
  
  mwSize num,sizeWm,sizeWn;
  double* u;
  mxArray * v;
  mwSize nzmaxNew,nzmaxOld;
  mwIndex *ir, *jc, *irs, *jcs;
  double *weights,*sr;
  mwIndex currentColumnIndex,currentEntryIndexOld, numColumnEntriesOld,currentRowIndex;
  mwIndex currentEntryIndexNew,skippedEntries;
  double ux,uy,derivative;
  void * newSr, * newIrs;
             
  // Test number of parameters.
  if (nrhs != 2 || nlhs != 1) {
    mexWarnMsgTxt("Usage: K = getSparseDerivativeMatrix(u,W)");
    return;
  }
  
  // Parse parameters
  num = mxGetM(prhs[0]);
  u = mxGetPr(prhs[0]);
   
  sizeWm= mxGetM(prhs[1]);
  sizeWn= mxGetN(prhs[1]);
    
  if (!(num==sizeWm) || !(sizeWm==sizeWn)) {
    mexWarnMsgTxt("Error: Dimensions of u and W do not match.");
    return;
  }
  if (!mxIsSparse(prhs[1])) {
    mexWarnMsgTxt("Error: Expects sparse matrix W.");
    return;
  }
    
  nzmaxOld = mxGetNzmax(prhs[1]); // number of nonzero elements of sparse matrix
  ir = mxGetIr(prhs[1]);
  jc = mxGetJc(prhs[1]);
  weights = mxGetPr(prhs[1]);
    
  // Allocate memory for output (sparse real matrix)
  v = mxCreateSparse(num, num, nzmaxOld/2, mxREAL);
  sr = mxGetPr(v);
  irs = mxGetIr(v);
  jcs = mxGetJc(v);
  
  currentColumnIndex=0,currentEntryIndexOld=0;
  currentEntryIndexNew=0;
  nzmaxNew=nzmaxOld/2;
  skippedEntries=0;
  jcs[0]=0;
 
  while(currentEntryIndexOld<nzmaxOld && currentColumnIndex<sizeWn)
  {
      numColumnEntriesOld=jc[currentColumnIndex+1];
      uy=u[currentColumnIndex];
       
      while(currentEntryIndexOld<numColumnEntriesOld)
      {
          currentRowIndex=ir[currentEntryIndexOld];
          
          if (currentRowIndex<currentColumnIndex)
          {
              ux= u[currentRowIndex];
              derivative=uy-ux;
              
              if (derivative != 0)
              {
                  sr[currentEntryIndexNew]=derivative;
                  irs[currentEntryIndexNew]=currentRowIndex;
                  currentEntryIndexNew++;
              }
              else
              {
                  skippedEntries++;
              }
          }
          else
          {
              skippedEntries++;
          }
          currentEntryIndexOld++;
      }
      jcs[currentColumnIndex+1]=numColumnEntriesOld-skippedEntries;
      
      currentColumnIndex++;
  }
  
  nzmaxNew=nzmaxOld-skippedEntries;
  
  for(;currentColumnIndex<sizeWn;currentColumnIndex++)
  {
      jcs[currentColumnIndex+1]=nzmaxNew;
  }
  
  newSr = mxRealloc(sr, nzmaxNew * sizeof(double));
  mxSetPr(v,(double*)newSr);
  newIrs= mxRealloc(irs, nzmaxNew * sizeof(mwIndex));
  mxSetIr(v,(mwIndex*) newIrs);
  mxSetNzmax(v,nzmaxNew);
  
  plhs[0]=v; 

}
