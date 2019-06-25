function [TrainEdgesZZ, TestPairsZZ] = CalculateZZ(Z,TrainEdges,TestPairs)

TrainEdgesZZ = Z(TrainEdges(:,1),:).*Z(TrainEdges(:,2),:);
TestPairsZZ = Z(TestPairs(:,1),:).*Z(TestPairs(:,2),:);
