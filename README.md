# Soft Affiliation Graph Model - A scalable overlapping community detection

Soft Affiliation Graph Model (S-AGM) is a soft community assignment model to detect overlapping communities with pluralistic homophily.
Stochastic Gradient Riemannian Langevin Dynamics (SGRLD) can be used to get scalable inference algorithm for S-AGM.

## Datasets
The small datasets used in Matlab proof-of-concept section are in
``
S-AGM/MatlabComparison/Datasets/.
``
The larger datasets used in SGRLD Tensorflow code are in 
``
S-AGM/TensorflowSGRLD/Datasets/.
``

## Codes
Matlab codes for proof-of-concept section are in 
``S-AGM/MatlabComparison/SmallNetworks/`` folder,
which contains MCMC for non-collapsed model of AGM, S-AGM and a-MMSB.
Matlab codes for comparison on synthetic networks between AGM and batch SGRLD for S-AGM are in 
``S-AGM/MatlabComparison/GeneratedNetworks/.``

Python codes with Tensorflow used for mini-batch SGRLD of S-AGM are in 
``S-AGM/TensorflowSGRLD/SGRLDMiniBatch/.``

The shell scripts to run the codes and plot figures in the paper 
are in ``S-AGM/MatlabComparison/ShellScripts/`` and ``S-AGM/TensorflowSGRLD/ShellScripts/`` folders .

## Running the codes

This contains required steps to get the results used in paper. After running the codes, the results are stored in Results folder.

### Prerequisites

Matlab R2017a 64-bits, Python 2.7.10 or above.

Required libraries for python: numpy-1.14.5, tensorflow-1.10.1, scipy-1.2.1. seaborn-0.9.0

### To run matlab codes, follow the steps 

```
cd S-AGM/MatlabComparison/ShellScripts/
chmod 777 ./*.sh
```

#### For small real world datasets (Football, NIPS234, Protein230)

To run AGM,

```
./GibbsAGMRealnetwork.sh 
```

To run a-MMSB,

```
./GibbsAMMSBRealnetwork.sh
```

To run S-AGM,

```
./GibbsSAGMRealnetwork.sh
```

After running AGM, a-MMSB and S-AGM, to produce plots in the paper,

```
./Smallnetworks-Processing-Result.sh
```

The plots (Fig. 1, 2, 3, 4 in the paper) will be stored in ``S-AGM/MatlabComparison/Plots/`` and Table 1 in ``table1.log`` in current folder.

#### For Synthetic networks from AGM generative process

To run AGM,
```
./GibbsAGMGeneratednetwork.sh
```

To run Batch SGRLD for S-AGM,
```
./SGRLDSAGMGeneratednetwork.sh
```

After running AGM and SGRLD for S-AGM on synthetic network, to produce plots in the paper
```
./Generatednetwork_NMI-Processing-Result.sh
./Generatednetwork_Scalability-Processing-Result.sh
```

The plots (Fig. 5, 6, 7, 8 in the paper) will be stored in ``S-AGM/MatlabComparison/Plots/``.

### To run tensorflow codes for mini-batch SGRLD for S-AGM and to produce plots in the paper

```
cd S-AGM/TensorflowSGRLD/ShellScripts/
chmod 777 ./*.sh
```

#### To run mini-batch SGRLD for S-AGM on larger real world datasets (FreeAssoc, Reuters, ca-HepPh)

To run mini-batch SGRLD for S-AGM,
```
./larger-graph.sh 
```
After running mini-batch SGRLD for S-AGM, to produce plots and Table 2 in the paper
```
./larger-graph-Processing-Result.sh 
```

The plots (Fig. 9 in the paper) are stored in ``S-AGM/TensorflowSGRLD/Plots/`` and Table 2 in table2.log in current folder.

#### To run mini-batch SGRLD for S-AGM on com-dblp network

To run mini-batch SGRLD for S-AGM,
```
./com-dblp.sh
```
After running mini-batch SGRLD for S-AGM, to produce plots in the paper,
```
./com-dblp-Processing-Result.sh 
```

The plots (Fig. 10 in the paper) are stored in ``S-AGM/TensorflowSGRLD/Plots/``.

#### To compare various mini-batches on ca-HepPh network

To run mini-batch SGRLD for S-AGM,
```
./ca-HepPh.sh
```

After running mini-batch SGRLD for S-AGM, to produce plots in the paper,
```
./ca-HepPh-Processing-Result.sh 
```
The plots (Fig. 11 in the paper) are stored in ``S-AGM/TensorflowSGRLD/Plots/``.

# OUTPUT
After running the codes, the outputs (plots) are stored in 
``S-AGM/MatlabComparison/Plots/`` for small networks from
Matlab codes and ``S-AGM/TensorflowSGRLD/Plots/`` for larger networks from Tensorflow codes.
