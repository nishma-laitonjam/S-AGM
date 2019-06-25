#!/usr/bin/env bash

# Figure 5 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Background_EdgeProb_nmiplot.m > ./output.log

# Figure 6 in paper
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Community_EdgeProb_nmiplot.m > ./output.log
