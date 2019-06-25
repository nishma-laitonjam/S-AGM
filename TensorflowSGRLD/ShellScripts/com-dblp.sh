#!/usr/bin/env bash
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'com-dblp' -o '../Results/' -l 4096 -m 32 -k 2048 -ns 100 -b 25000 -s 25000 -tr 0.1
