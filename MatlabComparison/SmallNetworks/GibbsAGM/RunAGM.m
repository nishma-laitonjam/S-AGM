% pass the file where it contains the network
function RunAGM(B, Outfile, K)
warning('off','all') % suppress the warnings

B = triu(B,1); % B is Upper Triangular Full Matrix(Train+Test)
clearvars -except B Outfile K;

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

A(TestIndex) = 1;
TrainNonEdgesFullMatrix = A + A';
TrainNonEdgesFullMatrix = 1 - TrainNonEdgesFullMatrix - eye(N);

TrainEdgeEachNodeIndex = cell(1,N);
TrainNonEdgeEachNodeIndex = cell(1,N);
% find the edge index and non edge index for each element
for i=1:N
    TrainEdgeEachNodeIndex{i} = find(TrainEdgesFullMatrix(:,i));%From 1 to N
    TrainNonEdgeEachNodeIndex{i} = find(TrainNonEdgesFullMatrix(:,i));
end
%%
[I, J] = find(triu(TrainEdgesFullMatrix,1));
TrainEdges = horzcat(I,J);
[I, J] = find(triu(TrainNonEdgesFullMatrix,1));
TrainNonEdges = horzcat(I,J);
clearvars TrainEdgesFullMatrix TrainNonEdgesFullMatrix A;

gamma0 = 1;
gamma1 = 1;

rho0 = 0.00005;
P0 = log(1-rho0); 
alpha = 1;

a = 5;
b = 1;
rho = betarnd(a, b, [K,1]);
P = log(1-rho);
Z = ones(N,K);

W = ones(1, K)*(1/K);
%% set the number of samples in NumIter and BurnIn in BurnIn
NumSamples = 2500; 
BurnIn = 2500;
T = 100;
%%
test_link = B(TestIndex);
tic;
[Sampled_Z, LLvector, ProbAve, Total_Probability_vs_num_shared_community, AUCvector,PerplexityVector, TimeVector] = AGMmcmc(TestPairs,TestIndex,test_link,Z,P,P0,W,alpha,gamma0,gamma1,a,b,BurnIn,NumSamples,TrainEdgeEachNodeIndex,TrainNonEdgeEachNodeIndex,TrainEdges,TrainNonEdges,T);
time = toc;
disp("The time taken");
disp(time);
disp(sum(Sampled_Z));
rate = ProbAve(TestIndex);
[~,~,~,AUCroc] = perfcurve(test_link,rate,1);
disp("AUCroc")
disp(AUCroc);
fprintf("Average Edge Probability");
disp(Total_Probability_vs_num_shared_community(1,:)./(Total_Probability_vs_num_shared_community(2,:)));

save(Outfile,'Sampled_Z','LLvector','ProbAve','time', 'Total_Probability_vs_num_shared_community','AUCvector', 'PerplexityVector', 'TimeVector', 'AUCroc');


