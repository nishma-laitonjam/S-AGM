#!/bin/bash

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAGM/RealNetworkEPMfootball.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAGM/RealNetworkEPMProtein230.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAGM/RealNetworkEPMnips17.m  > ./output.log

