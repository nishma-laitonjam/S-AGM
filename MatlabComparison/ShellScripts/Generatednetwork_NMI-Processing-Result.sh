#!/usr/bin/env bash

# Figure 1 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Background_EdgeProb_nmiplot.m > ./output.log

# Figure 2 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Community_EdgeProb_nmiplot.m > ./output.log
