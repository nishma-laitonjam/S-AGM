#!/bin/bash

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAMMSB/RealNetworkEPMfootball.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAMMSB/RealNetworkEPMProtein230.m  > ./output.log

nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../SmallNetworks/GibbsAMMSB/RealNetworkEPMnips17.m  > ./output.log
