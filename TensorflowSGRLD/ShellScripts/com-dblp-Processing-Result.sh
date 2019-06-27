#!/usr/bin/env bash

# Fig. 11: Trace plot of AUC-ROC score and perplexity of test data for com-dblp
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/com_dblp.m > ./output.out
