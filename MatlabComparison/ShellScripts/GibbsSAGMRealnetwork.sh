#!/bin/bash

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsSAGM/RealNetworkEPMfootball.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsSAGM/RealNetworkEPMProtein230.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsSAGM/RealNetworkEPMnips17.m  > ./output.log
