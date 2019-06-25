function [Probability_vs_num_shared_community] = Cal_Prob_vs_shared(Z,TrainEdges,TrainNonEdges)

Train_edge_num_shared_communities = sum(Z(TrainEdges(:,1),:).*Z(TrainEdges(:,2),:),2);
Train_non_edge_num_shared_communities = sum(Z(TrainNonEdges(:,1),:).*Z(TrainNonEdges(:,2),:),2);

K = size(Z,2);
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

