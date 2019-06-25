function [Z, n_i, m_k] = sample_Z_parallel(TrainEdges, TrainNonEdges, Z, n_i, m_k, K, pi, beta, epsilon)
N = size(Z,1);
Beta_Epsilon = ones(K)*epsilon;
Beta_Epsilon = Beta_Epsilon - diag(diag(Beta_Epsilon)) + diag(beta);

% TrainEdge
Z_train_edges = zeros(size(Z));
pi_TrainEdges_start = pi(TrainEdges(:,1),:);
pi_TrainEdges_end = pi(TrainEdges(:,2),:);
pi_TrainEdges_end_reshape = reshape(pi_TrainEdges_end, 1, size(TrainEdges,1), K);
P_TrainEdges = pi_TrainEdges_start'.*pi_TrainEdges_end_reshape;
PBeta_TrainEdges = reshape(Beta_Epsilon,K,1,K).*P_TrainEdges;
PBeta_TrainEdges_reshape = reshape(permute(PBeta_TrainEdges,[1,3,2]),K*K,size(TrainEdges,1));
PBeta_TrainEdges_reshape_cell = num2cell(log(PBeta_TrainEdges_reshape), 1);
TrainEdgeZ = cellfun(@logsample, PBeta_TrainEdges_reshape_cell);
[TrainEdgeZ_i, TrainEdgeZ_j] = ind2sub([K,K],TrainEdgeZ);
TrainEdge_i_j_index = sub2ind([N,N],TrainEdges(:,1),TrainEdges(:,2));
Z_train_edges(TrainEdge_i_j_index) = TrainEdgeZ_i;
TrainEdge_j_i_index = sub2ind([N,N],TrainEdges(:,2),TrainEdges(:,1));
Z_train_edges(TrainEdge_j_i_index) = TrainEdgeZ_j;

% TrainNonEdge
Z_train_non_edges = zeros(size(Z));
pi_TrainNonEdges_start = pi(TrainNonEdges(:,1),:);
pi_TrainNonEdges_end = pi(TrainNonEdges(:,2),:);
pi_TrainNonEdges_end_reshape = reshape(pi_TrainNonEdges_end, 1, size(TrainNonEdges,1), K);
P_TrainNonEdges = pi_TrainNonEdges_start'.*pi_TrainNonEdges_end_reshape;
PBeta_TrainNonEdges = reshape(1-Beta_Epsilon,K,1,K).*P_TrainNonEdges;
PBeta_TrainNonEdges_reshape = reshape(permute(PBeta_TrainNonEdges,[1,3,2]),K*K,size(TrainNonEdges,1));
PBeta_TrainNonEdges_reshape_cell = num2cell(log(PBeta_TrainNonEdges_reshape), 1);
TrainNonEdgeZ = cellfun(@logsample, PBeta_TrainNonEdges_reshape_cell);
[TrainNonEdgeZ_i, TrainNonEdgeZ_j] = ind2sub([K,K],TrainNonEdgeZ);
TrainNonEdge_i_j_index = sub2ind([N,N],TrainNonEdges(:,1),TrainNonEdges(:,2));
Z_train_non_edges(TrainNonEdge_i_j_index) = TrainNonEdgeZ_i;
TrainNonEdge_j_i_index = sub2ind([N,N],TrainNonEdges(:,2),TrainNonEdges(:,1));
Z_train_non_edges(TrainNonEdge_j_i_index) = TrainNonEdgeZ_j;


Z = Z_train_edges + Z_train_non_edges;
for k = 1:K
    % Update m_k for train edges
    m_k(k,1) = sum(sum(triu((Z_train_edges==k) .* (Z_train_edges'==k))));
    
    % Update m_k for train non edges
    m_k(k,2) = sum(sum(triu((Z_train_non_edges==k) .* (Z_train_non_edges'==k))));
    
    % Update n_i
    n_i(:, k) = sum(Z==k,2);
end