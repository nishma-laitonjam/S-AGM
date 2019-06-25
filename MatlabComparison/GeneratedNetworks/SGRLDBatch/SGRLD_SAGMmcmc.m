function [Sampled_w,LLvector,Avg_TestPairProb, AUCvector,PerplexityVector,TimeVector]= SGRLD_SAGMmcmc(NumSamples, NumBurnIn, TrainEdges, TrainNonEdges, StepSizeWA, StepSizePiA, StepSizeB, StepSizeC, alpha_shape, alpha_rate, eta_0, eta_1, alpha, Pi0, Pi, SumPiBar, w, SumWBar, TestPairs, TestLinks, FullB, FullBTrainEdgeIndex, WMasksFull, T)

P0 = log(1-Pi0);
TotalIter = NumSamples + NumBurnIn;
K = size(w,2);
N = size(w,1);
disp(K);
disp(N);
TotalTestPairProb = zeros(size(TestPairs,1),1);

AUCvector = [];
TimeVector = [];
PerplexityVector = [];
LLvector = [];
if TotalIter > 0
    for iter = 1:NumBurnIn
        iter_tic = tic;
        StepSizeW  = StepSizeWA*((1 + iter/StepSizeB)^(-StepSizeC));
        StepSizePi  = StepSizePiA*((1 + iter/StepSizeB)^(-StepSizeC));
        
        % Sample WBar
        [w, SumWBar] = Sample_w(StepSizeW, alpha, Pi0, Pi, w, SumWBar, N, K, FullB, FullBTrainEdgeIndex, WMasksFull);
        
        % Sample PiBar
        [Pi, SumPiBar] = Sample_pi(TrainEdges, TrainNonEdges, StepSizePi, eta_0, eta_1, Pi0, Pi, SumPiBar, w, K);

        % Sample alpha
        alpha = Sample_alpha(alpha_shape, alpha_rate, w, N);
        
        % Compute Log Likelihood of training data
        LLTrainNonEdge =  P0 + sum(log(1-w(TrainNonEdges(:,1),:).*w(TrainNonEdges(:,2),:).*Pi),2);
        LLTrainEdge = log(1-exp(P0 + sum(log(1-w(TrainEdges(:,1),:).*w(TrainEdges(:,2),:).*Pi),2)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];
        
        
        % Print current iteration
        if mod(iter,T)==0
            fprintf("Number of iterations = %d\n",iter);
            % calculate AUC-ROC without sampling Z
            StartNode_prob = w(TestPairs(:,1),:);
            EndNode_prob = w(TestPairs(:,2),:);
            LogSumTest = P0 + sum(log(1-StartNode_prob.*EndNode_prob.*Pi),2);
            CurrTestPairProb = 1-exp(LogSumTest);
            [~,~,~,AUCroc] = perfcurve(TestLinks,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(LogSumTest(TestLinks==0),1)+sum(log(1-exp(LogSumTest(TestLinks==1))),1);
            Perp = exp(-(SumTestLL)/size(TestLinks,1));
            PerplexityVector = [PerplexityVector; Perp];
        end
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    for iter = (NumBurnIn+1):(NumBurnIn+NumSamples)
        iter_tic = tic;
        StepSizeW  = StepSizeWA*((1 + iter/StepSizeB)^(-StepSizeC));
        StepSizePi  = StepSizePiA*((1 + iter/StepSizeB)^(-StepSizeC));
        
        % Sample WBar
        [w, SumWBar] = Sample_w(StepSizeW, alpha, Pi0, Pi, w, SumWBar, N, K, FullB, FullBTrainEdgeIndex, WMasksFull);
        
        % Sample PiBar
        [Pi, SumPiBar] = Sample_pi(TrainEdges, TrainNonEdges, StepSizePi, eta_0, eta_1, Pi0, Pi, SumPiBar, w, K);

        % Sample alpha
        alpha = Sample_alpha(alpha_shape, alpha_rate, w, N);
        
        % Compute Log Likelihood of training data
        LLTrainNonEdge =  P0 + sum(log(1-w(TrainNonEdges(:,1),:).*w(TrainNonEdges(:,2),:).*Pi),2);
        LLTrainEdge = log(1-exp(P0 + sum(log(1-w(TrainEdges(:,1),:).*w(TrainEdges(:,2),:).*Pi),2)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];
        
        % Print current iteration
        StartNode_prob = w(TestPairs(:,1),:);
        EndNode_prob = w(TestPairs(:,2),:);
        LogSumTest = P0+sum(log(1-StartNode_prob.*EndNode_prob.*Pi),2);
        CurrTestPairProb = 1-exp(LogSumTest);
        if mod(iter,T)==0
            fprintf("Number of iterations = %d\n",iter);
            % calculate AUC-ROC without sampling 
            [~,~,~,AUCroc] = perfcurve(TestLinks,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(LogSumTest(TestLinks==0),1)+sum(log(1-exp(LogSumTest(TestLinks==1))),1);
            Perp = exp(-(SumTestLL)/size(TestLinks,1));
            PerplexityVector = [PerplexityVector; Perp];
        end
        % output data
        TotalTestPairProb = TotalTestPairProb + CurrTestPairProb;
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    Avg_TestPairProb = TotalTestPairProb/NumSamples;
    Sampled_w = w;
end
