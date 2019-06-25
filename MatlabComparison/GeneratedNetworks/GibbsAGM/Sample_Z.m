function [Z] = Sample_Z(TrainEdgeEachNodeIndex,TrainNonEdgeEachNodeIndex, W, P, P0, Z, N, K)
for i=1:N
    EdgeIndex = TrainEdgeEachNodeIndex{i};
    NumNonEdge = sum(Z(TrainNonEdgeEachNodeIndex{i},:),1);
    for k=1:K
        tempZ = repmat(Z(i,:),2,1);
        tempZ(1,k) = 0;
        tempZ(2,k) = 1;

        EdgeZPZ = tempZ.*P'*Z(EdgeIndex,:)' + P0;
        Edge = log(1-exp(EdgeZPZ));

        NonEdgeZPZ = NumNonEdge(k)*P(k);
        NonEdge = [0;NonEdgeZPZ];

        logP_X = sum(Edge,2)+ NonEdge +[log(1-W(k));log(W(k))];
        Z(i,k) = logsample(logP_X)-1;
    end
end