function RunSGRLD_AGM_MMSB(B, Outfile, K, GroundTruth)
warning('off','all') % suppress the warnings
B = triu(B,1);
clearvars -except B Outfile K GroundTruth;

%% Find the Test Pairs
TrainRatio = 0.8;
[~,TestIndex,~] = Create_Mask_network(B, TrainRatio);% stores only the TestIndex
[TestI, TestJ] = ind2sub(size(B),TestIndex);
TestPairs = horzcat(TestI,TestJ);% stores test pairs
TestLinks = B(TestIndex);
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
[TrainNonEdgesJ,TrainNonEdgesI] = find(tril(TrainNonEdgesFullMatrix,-1));
TrainNonEdges = horzcat(TrainNonEdgesI,TrainNonEdgesJ);

B(TestIndex) = 0;
FullB = B' + B; % sparse
FullIndex = find(FullB);
[FullTrainEdgeRow, FullTrainEdgeCol] = find(FullB);
FullBTrainEdgeIndex = horzcat(FullIndex, FullTrainEdgeRow, FullTrainEdgeCol);
WMaskIndex = sub2ind(size(B), [TestPairs(:,1);TestPairs(:,2);(1:N)'], [TestPairs(:,2);TestPairs(:,1);(1:N)']);
WMasksFull = zeros(N);
WMasksFull(WMaskIndex) = 1;
WMasksFull(FullIndex) = 2;
WMasksFull = 1 - WMasksFull;

clearvars TrainEdgesFullMatrix TrainNonEdgesFullMatrix A B FullIndex FullTrainEdgeRow FullTrainEdgeCol WMaskIndex;
%% Initialize the variables
alpha_shape = 1;
alpha_rate = 1;
eta_0 = 5;
eta_1 = 1;
alpha = ones(1,K);
disp(alpha)

Pi0 = 0.00005;
PiBarGammaShape = repmat([eta_0; eta_1],1,K);
PiBar = gamrnd(PiBarGammaShape,1);

StepSizeWA = 0.01;
StepSizePiA = 0.01;
StepSizeB = 1000;
StepSizeC = 0.55;

SumWBar = ones(N,K);
w = ones(N,K)*(1/K);%changed
SumPiBar = sum(PiBar,1);
Pi = PiBar(1,:)./SumPiBar;%changed

clearvars WBar PiBar;
%% Run SGRLD AGM MMSB
NumBurnIn = 500;
NumSamples = 500;
T = 100;

start_time = tic;
[Sampled_w, LLvector, Avg_TestPairProb, AUCvector, PerplexityVector, TimeVector] ...
    = SGRLD_SAGMmcmc(NumSamples, NumBurnIn, TrainEdges, TrainNonEdges, StepSizeWA, StepSizePiA, StepSizeB, StepSizeC, alpha_shape, alpha_rate, eta_0, eta_1, alpha, Pi0, Pi, SumPiBar, w, SumWBar, TestPairs,TestLinks, FullB, FullBTrainEdgeIndex, WMasksFull, T);
if exist('GroundTruth','var')
    % compute NMI with various threshold (0.5, 0.6, 0.7, 0.8, 0.9)
    threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    nmi = [];
    [GTcurrI, GTcurrJ] = find(GroundTruth);
    for i = 1:size(GroundTruth, 2)
        GTCommZ{i} = GTcurrI(GTcurrJ == i)';
    end
    for curr_threshold = threshold
        Z = Sampled_w>curr_threshold;
        [currI, currJ] = find(Z);
        for i = 1:size(Z,2)
            CommZ{i} = currI(currJ== i)';
        end
        nmi = [nmi, mutual3(CommZ, GTCommZ)];
    end
    disp(nmi);
end
stop_time = toc(start_time);
disp("The time taken");
disp(stop_time);
[~,~,~,AUCroc] = perfcurve(TestLinks,Avg_TestPairProb,1);
disp("AUC-ROC");
disp(AUCroc);

%% Save the outputs "Test_edge_log_prob, total_pi"
save(Outfile,'Sampled_w','nmi','Avg_TestPairProb','stop_time', 'AUCroc', 'AUCvector', 'TimeVector','PerplexityVector','LLvector');

