function P = Sample_pi(eta_0,eta_1,P,P0,CountTrainNonEdgeEachCluster, TrainEdgeZZ,NumTrainNonEdges)

[P] = UpdateParametersHMC(CountTrainNonEdgeEachCluster, TrainEdgeZZ,P,P0,eta_0,eta_1,NumTrainNonEdges);
