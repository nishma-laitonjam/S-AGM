#!/bin/bash

## Changing Network size N
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/GibbsAGM/GeneratedChangeN.m  > output.log

## Changing Background Edge Probability
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/GibbsAGM/GeneratedBackgroundProb.m  > output.log

## Changing Community Edge Probability
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/GibbsAGM/GeneratedCommunityProb.m  > output.log
