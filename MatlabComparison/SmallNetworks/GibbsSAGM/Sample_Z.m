function [z_ik, CountTrainNonEdgeEachCluster, TrainEdgeZZ, Probability_vs_num_shared_community] = Sample_Z(TrainEdges,TrainNonEdges,N,K,w,P,P0,TrainEdgeZZ)
ProbableZ = [0 1 0 1; 1 0 0 1; 0 1 1 0; 1 0 1 0]; % z_ijk, 1- z_ijk, z_jik, 1 - z_jik
CountTrainNonEdgeEachCluster = zeros(1,K);
Transpose_w = w';
z_ik = zeros(N,K);
Train_edge_num_shared_communities = zeros(1, size(TrainEdges,1));
Train_non_edge_num_shared_communities = zeros(1, size(TrainNonEdges,1));
for k = 1:K
    % Train Edges
    FirstNodeTrainEdge = TrainEdges(:,1);
    SecondNodeTrainEdge = TrainEdges(:,2);
    ProbableZTrainEdge = ProbableZ*[log(Transpose_w(k,FirstNodeTrainEdge)); log(1-Transpose_w(k,FirstNodeTrainEdge)); log(Transpose_w(k,SecondNodeTrainEdge)); log(1-Transpose_w(k,SecondNodeTrainEdge))];
    WithoutCurrK_P = P;
    WithoutCurrK_P(k) = 0;
    WithoutCurrKTerm = sum(WithoutCurrK_P.*TrainEdgeZZ,2)';
    TrainEdgeWithK = log(1-exp(P0+P(k)+WithoutCurrKTerm));
    TrainEdgeWithoutK = log(1-exp(P0+WithoutCurrKTerm));
    TrainEdgeLogLikelihood =  ProbableZTrainEdge + [TrainEdgeWithoutK;TrainEdgeWithoutK;TrainEdgeWithoutK;TrainEdgeWithK];
    % Sample assignments for Train Edges
    TrainEdgeAssignment = cellfun(@logsample, num2cell(TrainEdgeLogLikelihood, 1)); %every column
    TrainEdgeAssignmentIndex = de2bi(TrainEdgeAssignment-1,2);% 0,1,2,3
    Train_edge_num_shared_communities(TrainEdgeAssignment==4) = Train_edge_num_shared_communities(TrainEdgeAssignment==4) + 1;% Update Train_edge_num_shared_communities
    % Update TrainEdgeZZ
    TrainEdgeZZ(:,k) = TrainEdgeAssignmentIndex(:,1).*TrainEdgeAssignmentIndex(:,2);
    % Train Non Edges
    FirstNodeTrainNonEdge = TrainNonEdges(:,1);
    SecondNodeTrainNonEdge = TrainNonEdges(:,2);
    TrainNonEdgeLogLikelihood  = (ProbableZ*[log(Transpose_w(k,FirstNodeTrainNonEdge)); log(1-Transpose_w(k,FirstNodeTrainNonEdge)); log(Transpose_w(k,SecondNodeTrainNonEdge)); log(1-Transpose_w(k,SecondNodeTrainNonEdge))])'+[0,0,0,P(k)];
    % Sample assignments for Train Non Edges
    TrainNonEdgeAssignment = cellfun(@logsample, num2cell(TrainNonEdgeLogLikelihood', 1)); %every column
    TrainNonEdgeAssignmentIndex = de2bi(TrainNonEdgeAssignment-1,2);% 0,1,2,3
    Train_non_edge_num_shared_communities(TrainNonEdgeAssignment==4) = Train_non_edge_num_shared_communities(TrainNonEdgeAssignment==4) + 1; % Update Train_non_edge_num_shared_communities
    % Update CountTrainNonEdgeEachCluster
    CountTrainNonEdgeEachCluster(k) = sum(TrainNonEdgeAssignment==4);
    % Find z_ik
    NodeIndexCurrK = [TrainEdges.*TrainEdgeAssignmentIndex;TrainNonEdges.*TrainNonEdgeAssignmentIndex];
    CountNodeCurrK = histc(NodeIndexCurrK(:),1:N);
    z_ik(:,k) = CountNodeCurrK;
end
Count_shared_community = zeros(2,K+1);
for k = 1:(K+1)
    Count_shared_community(1,k) = sum(Train_edge_num_shared_communities == (k-1));
    Count_shared_community(2,k) = sum(Train_non_edge_num_shared_communities == (k-1));
end
Probability_vs_num_shared_community = ones(2, (K+1));
Probability_vs_num_shared_community(1,:) = Count_shared_community(1,:)./sum(Count_shared_community,1);
[~, col] = find(isnan(Probability_vs_num_shared_community));
Probability_vs_num_shared_community(1,col) = 0;
Probability_vs_num_shared_community(2,col) = 0;
