#!/usr/bin/env bash

# Fig. 10: Trace plot of AUC-ROC score and perplexity of test data for ca-HepPh
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/caHepPh.m > ./output.out
