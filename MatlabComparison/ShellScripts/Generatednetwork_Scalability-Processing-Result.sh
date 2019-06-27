#!/usr/bin/env bash

# Figure 3 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/scalability_aucplot.m > ./output.log

# Figure 4 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/scalability_timeplot.m > ./output.log
