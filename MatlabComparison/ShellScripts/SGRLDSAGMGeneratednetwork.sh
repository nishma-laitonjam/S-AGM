#!/bin/bash

## Changing Network size N
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/SGRLDBatch/GeneratedChangeN.m  > output.log

## Changing Background Edge Probability
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/SGRLDBatch/GeneratedBackgroundProb.m  > output.log

## Changing Community Edge Probability
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../GeneratedNetworks/SGRLDBatch/GeneratedCommunityProb.m  > output.log
