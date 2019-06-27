#!/usr/bin/env bash

## Fig. 9: Trace plot of perplexity of test data for various K
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/TestPP_Larger_Graph.m > ./output.out

## Table 2: AUC-ROC scores of test data and runtime (hrs) for various K
python -W ignore ../ProcessingResult/Table2.py >table2.out
