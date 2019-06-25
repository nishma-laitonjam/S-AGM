#!/usr/bin/env bash

## Plot for perplexity
nohup /Applications/MATLAB_R2017a.app/bin/matlab -nodisplay < ../ProcessingResult/TestPP_Larger_Graph.m > ./output.out

## Plot the table2
python -W ignore ../ProcessingResult/Table2.py >table2.out
