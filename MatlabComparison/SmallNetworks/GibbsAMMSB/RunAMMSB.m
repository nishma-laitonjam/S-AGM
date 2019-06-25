% pass the file where it contains the network
function RunAMMSB(B, Outfile, K, GroundTruthFile)

if nargin<4
    GroundTruth = [];
else
    GroundTruth = readGroundTruth(GroundTruthFile);
end


B = triu(B,1); % B is Upper Triangular Full Matrix(Train+Test)
clearvars -except B Outfile K GroundTruth;

%% Find the Test Pairs
TrainRatio = 0.8;
[~,TestIndex,~] = Create_Mask_network(B, TrainRatio);% stores only the TestIndex
[TestI, TestJ] = ind2sub(size(B),TestIndex);
TestPairs = horzcat(TestI,TestJ);% stores test pairs
TestIndex_LTM = sub2ind(size(B), TestJ, TestI);
%%
N = size(B,1);
B_NonEdge = triu(1-B,1); % Upper Triangular Full Matrix(Train+Test) where edge = 0 and non-edge = 1

A = B;
A(TestIndex) = 0;
TrainEdgesIndex = find(A);
[TrainEdgesI,TrainEdgesJ] = ind2sub(size(B),TrainEdgesIndex);
TrainEdges = horzcat(TrainEdgesI,TrainEdgesJ);

A = B_NonEdge;
A(TestIndex) = 0;
TrainNonEdgesIndex = find(A);
[TrainNonEdgesJ,TrainNonEdgesI] = ind2sub(size(B),TrainNonEdgesIndex);
TrainNonEdges = horzcat(TrainNonEdgesI,TrainNonEdgesJ);%ADDED

test_link = B(TestIndex);

%initialization
alpha = ones(1,K)*0.1;
epsilon = 0.00005;
a = 5;
b = 1;

Z = ones(N,N)*K;
Z = Z - diag(diag(Z));
Z(TestIndex) = 0;
Z(TestIndex_LTM) = 0;
n_i = find_n_i(Z, N, K);
m_k = find_m_k(Z, K, B, B_NonEdge);
pi = ones(N, K)*(1/K);
beta = betarnd(a, b, [K, 1]);

clearvars -except K Outfile test_link TrainEdges TrainNonEdges TestPairs Z n_i m_k pi beta a b alpha epsilon;
%% set the number of samples in NumIter and BurnIn in BurnIn
NumSamples = 2500; 
BurnIn = 2500;
T = 100;
%%
tic;
[Sampled_Z, ProbAve, Sampled_pi, Avg_pi, Avg_beta, Sampled_n_i, AUCvector, PerplexityVector, LLvector, TimeVector] = AMMSBmcmc(test_link, NumSamples, BurnIn, TrainEdges, TrainNonEdges, TestPairs, Z, n_i, m_k, pi, beta, a, b, alpha, epsilon, T);
time = toc;
disp("The time taken");
disp(time);
CommAssign = FindCommAssign(Sampled_Z, K);
%disp(CommAssign);
rate = ProbAve;

[~,~,~,AUCroc] = perfcurve(test_link,rate,1);
disp("AUCroc")
disp(AUCroc);

%disp(LLvector')
save(Outfile,'Sampled_Z','ProbAve','Sampled_pi','Avg_pi','Avg_beta','Sampled_n_i','time','AUCroc','CommAssign', 'AUCvector', 'PerplexityVector', 'LLvector', 'TimeVector');
