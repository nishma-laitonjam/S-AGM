#!/usr/bin/env bash

# Figure 7 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/scalability_aucplot.m > ./output.log

# Figure 8 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/scalability_timeplot.m > ./output.log
