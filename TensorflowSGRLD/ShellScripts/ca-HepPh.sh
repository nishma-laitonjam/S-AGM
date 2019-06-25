#!/usr/bin/env bash
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch_time.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 5 -m 5 -k 50 -ns 100 -tr 0.1 -mt 1801 -t 30
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch_time.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 100 -m 100 -k 50 -ns 100 -tr 0.1 -mt 1801 -t 30
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch_time.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 500 -m 500 -k 50 -ns 100 -tr 0.1 -mt 1801 -t 30
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch_time.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 1000 -m 1000 -k 50 -ns 100 -tr 0.1 -mt 1801 -t 30
