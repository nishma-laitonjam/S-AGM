function [lpdf,glpdf] = Dist(r,eta_0,eta_1,CountTrainNonEdgeEachCluster,TrainEdgeZZ,P0,NumTrainNonEdges)

pi = 1./(1+exp(-r));
P = log(1-pi);

TrainEdgeZZp = TrainEdgeZZ.*P;
TrainEdgeZPTrainEdgeZ = sum(TrainEdgeZZp,2)+P0;

TrainEdge = sum(log(1-exp(TrainEdgeZPTrainEdgeZ)),1);

TrainNonEdge = sum(CountTrainNonEdgeEachCluster.*P,2)+P0*NumTrainNonEdges;

lpdf = TrainEdge + TrainNonEdge + sum(eta_0.*r-(eta_0+eta_1).*log(1+exp(r)));

ZZpi = TrainEdgeZZ.*pi;

TrainEdge = sum(exp(TrainEdgeZPTrainEdgeZ)./(1-exp(TrainEdgeZPTrainEdgeZ)).*ZZpi,1);% row vector

TrainNonEdge = -CountTrainNonEdgeEachCluster.*pi; % row vector

glpdf = TrainEdge + TrainNonEdge -(eta_0+eta_1).*pi+eta_0; % row vector