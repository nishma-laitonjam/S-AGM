#!/usr/bin/env bash

## ca-HepPh
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 1000 -m 1000 -k 50 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 1000 -m 1000 -k 100 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 1000 -m 1000 -k 150 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'ca-HepPh' -o '../Results/' -l 1000 -m 1000 -k 200 -ns 100 -b 2500 -s 2500 -tr 0.1

## FreeAssoc
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'FA' -o '../Results/' -l 1000 -m 1000 -k 50 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'FA' -o '../Results/' -l 1000 -m 1000 -k 100 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'FA' -o '../Results/' -l 1000 -m 1000 -k 150 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'FA' -o '../Results/' -l 1000 -m 1000 -k 200 -ns 100 -b 2500 -s 2500 -tr 0.1

## Reuters
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'Reuters' -o '../Results/' -l 1000 -m 1000 -k 50 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'Reuters' -o '../Results/' -l 1000 -m 1000 -k 100 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'Reuters' -o '../Results/' -l 1000 -m 1000 -k 150 -ns 100 -b 2500 -s 2500 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -n 'Reuters' -o '../Results/' -l 1000 -m 1000 -k 200 -ns 100 -b 2500 -s 2500 -tr 0.1
