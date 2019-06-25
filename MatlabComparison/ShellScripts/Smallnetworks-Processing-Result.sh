#!/usr/bin/env bash

# Figure 1 in paper (Pluralistic Homophily)
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/PluralisticHomophily.m > ./output.log

# Figure 2 and 3 in paper
python ../ProcessingResult/TracePlots.py > ./output.log

# Figure 4 in paper (Football network)
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Football.m > ./output.log

# Table 1 in paper 
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/RealnetworkAucTable.m > ./table1.log
