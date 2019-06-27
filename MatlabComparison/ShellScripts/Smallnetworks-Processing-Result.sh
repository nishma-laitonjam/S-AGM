#!/usr/bin/env bash

# Fig. 5: Pluralistic Homophily by AGM and S-AGM
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/PluralisticHomophily.m > ./output.log

# Figure 6 and 7 in paper (Trace plots)
python ../ProcessingResult/TracePlots.py > ./output.log

# Fig. 8: Communities for Football network.
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/Football.m > ./output.log

# Table 1: AUC-ROC 
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/RealnetworkAucTable.m > ./table1.log
