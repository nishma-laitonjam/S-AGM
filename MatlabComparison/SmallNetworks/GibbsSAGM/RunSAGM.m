function RunSAGM(B, Outfile, K)
warning('off','all') % suppress the warnings
B = triu(B,1);
clearvars -except B Outfile K GroundTruth;

%% Find the Test Pairs
TrainRatio = 0.8;
[~,TestIndex,~] = Create_Mask_network(B, TrainRatio);% stores only the TestIndex
[TestI, TestJ] = ind2sub(size(B),TestIndex);
TestPairs = horzcat(TestI,TestJ);% stores test pairs
%%
N = size(B,1);

A = B;
A(TestIndex) = 0;
TrainEdgesFullMatrix = A + A';
[TrainEdgesJ,TrainEdgesI] = find(tril(TrainEdgesFullMatrix,-1));
TrainEdges = horzcat(TrainEdgesI,TrainEdgesJ);

A(TestIndex) = 1;
TrainNonEdgesFullMatrix = A + A';
TrainNonEdgesFullMatrix = 1 - TrainNonEdgesFullMatrix - eye(N);
[TrainNonEdgesJ,TrainNonEdgesI] = find(tril(TrainNonEdgesFullMatrix,-1));%ADDED
TrainNonEdges = horzcat(TrainNonEdgesI,TrainNonEdgesJ);%ADDED

TrainFullMatrix = TrainEdgesFullMatrix+TrainNonEdgesFullMatrix;
CountTestEachNode = N-1-(sum(TrainFullMatrix,2));

NumTrainEdges = sum(sum(TrainEdgesFullMatrix))/2; % number of edges without test
NumTrainNonEdges = sum(sum(TrainNonEdgesFullMatrix))/2; % number of non edges without test
clearvars TrainFullMatrix TrainEdgesFullMatrix TrainNonEdgesFullMatrix A;
%% Initialize the variables
alpha_shape = 1;
alpha_rate = 1;
eta_0 = 5;
eta_1 = 1;
alpha = ones(1,K);
w = ones(N,K)*(1/K);
%w = betarnd(repmat(alpha,N,1),1);
pi0 = 0.00005;
P0 = log(1-pi0);
pi = betarnd(eta_0, eta_1, [1,K]);
P = log(1-pi);
TrainEdgeZZ = ones(NumTrainEdges,K);

%% Run Non Collapsed AGM MMSB
NumSamples = 2500;
NumBurnIn = 2500;
T = 100;

test_link = B(TestIndex);
tic;
[Avg_TestPairProb, Avg_w, Avg_pi, Sampled_Z, Total_Probability_vs_num_shared_community, AUCvector, PerplexityVector, LLvector, TimeVector]= SAGMmcmc(TrainEdges,TrainNonEdges,TrainEdgeZZ,eta_0,eta_1,alpha,alpha_shape,alpha_rate,P0,pi,P,w,NumSamples,NumBurnIn,CountTestEachNode,NumTrainNonEdges,TestPairs,test_link,T);
time = toc;
disp("The time taken");
disp(time);
[~,~,~,AUCroc] = perfcurve(test_link,Avg_TestPairProb,1);
disp("AUC-ROC");
disp(AUCroc);
fprintf("Average Edge Probability");
disp(Total_Probability_vs_num_shared_community(1,:)./(Total_Probability_vs_num_shared_community(2,:)));



%% Save the outputs "Test_edge_log_prob, total_pi"
save(Outfile,'Avg_TestPairProb','Avg_w','Avg_pi','time','AUCroc', 'Sampled_Z', 'Total_Probability_vs_num_shared_community', 'AUCvector', 'PerplexityVector', 'LLvector');

