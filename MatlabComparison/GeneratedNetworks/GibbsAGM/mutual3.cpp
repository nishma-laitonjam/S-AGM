#include "mutual3_standard_package/standard_include.cpp"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    #define one_in prhs[0]
    #define two_in prhs[1]
    
    deque<deque<int>> one_deque;
    deque<int> temp_deque;
    mxArray *one;
    double *one_d;
    const mwSize *dims; 
    dims = mxGetDimensions(one_in);
    for (int i=0; i<dims[1]; i++) {
        temp_deque.clear();
        one = mxGetCell(one_in,i);
        one_d = mxGetPr(one);
        for(int j=0;j<mxGetN(one);j++){
            temp_deque.push_back((int)one_d[j]);
        }
        sort(temp_deque.begin(), temp_deque.end());
        one_deque.push_back(temp_deque);
    }
    
    deque<deque<int>> two_deque;
    mxArray *two;
    double *two_d;
    dims = mxGetDimensions(two_in);
    for (int i=0; i<dims[1]; i++) {
        temp_deque.clear();
        two = mxGetCell(two_in,i);
        two_d = mxGetPr(two);
        for(int j=0;j<mxGetN(two);j++){
            temp_deque.push_back((int)two_d[j]);
        }
        sort(temp_deque.begin(), temp_deque.end());
        two_deque.push_back(temp_deque);
    }
    
    double a = mutual3(one_deque, two_deque);
    /***/
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *nmi = mxGetPr(plhs[0]);
    nmi[0] = a;
    return;
}

