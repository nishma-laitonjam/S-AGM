function [Sampled_Z, LLvector, ProbAve,Total_Probability_vs_num_shared_community,AUCvector,PerplexityVector,TimeVector] = AGMmcmc(TestPairs,TestIndex,test_link,Z,P,P0,W,alpha,gamma0,gamma1,a,b,BurnIn,NumSamples,TrainEdgeEachNodeIndex,TrainNonEdgeEachNodeIndex,TrainEdges,TrainNonEdges,T)

N = size(Z,1);
K = size(Z,2);
NumTestLink = size(test_link,1);

Total_Probability_vs_num_shared_community = zeros(2,K+1);

TrainEdgesIndex = sub2ind([N,N],TrainEdges(:,1),TrainEdges(:,2)); 
TrainNonEdgesIndex = sub2ind([N,N],TrainNonEdges(:,1),TrainNonEdges(:,2));
TrainNumNonEdges = size(TrainNonEdgesIndex,1);%N(N-1)/2 - (Y1 + Y?)

ProbAve = zeros(N,N);

TotalIter = NumSamples + BurnIn;

AUCvector = [];
PerplexityVector = [];
LLvector = [];
TimeVector = [];
if TotalIter > 0
    for iter=1:BurnIn
        iter_tic = tic;
        
        % Sample Z
        Z = Sample_Z(TrainEdgeEachNodeIndex,TrainNonEdgeEachNodeIndex, W, P, P0, Z, N, K);
        m = sum(Z,1);
        
        % Sample P
        [TrainEdgesZZ, TestPairsZZ] = CalculateZZ(Z,TrainEdges,TestPairs);
        P = UpdateParametersHMC(TestPairsZZ,TrainEdgesZZ,m,P,P0,a,b,TrainNumNonEdges);
        
        % Sample W
        W = Sample_W(alpha, m, N);
        
        % Sample alpha
        alpha = Sample_Alpha(gamma0, gamma1, K, W);
        % Output
        LogSum = (Z.*P')*Z'+P0;
        EdgeProb = 1-exp(LogSum);
        logP_X = sum(log(1-EdgeProb(TrainNonEdgesIndex))) + sum(log(EdgeProb(TrainEdgesIndex)));
        LLvector = [LLvector; logP_X];
        if mod(iter,T)==0
            disp(iter);
            CurrTestPairProb = EdgeProb(TestIndex);
            [~,~,~,AUCroc] = perfcurve(test_link,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(log(CurrTestPairProb(test_link==1)),1)+sum(log(1-CurrTestPairProb(test_link==0)),1);
            Perp = exp(-(SumTestLL)/NumTestLink);
            PerplexityVector = [PerplexityVector; Perp];
        end
        %Total_Probability_vs_num_shared_community = Total_Probability_vs_num_shared_community + Probability_vs_num_shared_community;
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    for iter=(BurnIn+1):(BurnIn+NumSamples)
        iter_tic = tic;
        
        % Sample Z
        Z = Sample_Z(TrainEdgeEachNodeIndex,TrainNonEdgeEachNodeIndex, W, P, P0, Z, N, K);
        m = sum(Z,1);
        [Probability_vs_num_shared_community] = Cal_Prob_vs_shared(Z,TrainEdges,TrainNonEdges);

        % Sample P
        [TrainEdgesZZ, TestPairsZZ] = CalculateZZ(Z,TrainEdges,TestPairs);
        P = UpdateParametersHMC(TestPairsZZ,TrainEdgesZZ,m,P,P0,a,b,TrainNumNonEdges);
        
        % Sample W
        W = Sample_W(alpha, m, N);
        
        % Sample alpha
        alpha = Sample_Alpha(gamma0, gamma1, K, W);
        % Output
        LogSum = (Z.*P')*Z'+P0;
        EdgeProb = 1-exp(LogSum);
        logP_X = sum(log(EdgeProb(TrainEdgesIndex))) + sum(log(1-EdgeProb(TrainNonEdgesIndex)));
        LLvector = [LLvector; logP_X];
        if mod(iter,T)==0
            disp(iter);
            CurrTestPairProb = EdgeProb(TestIndex);
            [~,~,~,AUCroc] = perfcurve(test_link,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(log(CurrTestPairProb(test_link==1)),1)+sum(log(1-CurrTestPairProb(test_link==0)),1);
            Perp = exp(-(SumTestLL)/NumTestLink);
            PerplexityVector = [PerplexityVector; Perp];
        end
        Total_Probability_vs_num_shared_community = Total_Probability_vs_num_shared_community + Probability_vs_num_shared_community;
        ProbAve = ProbAve + EdgeProb;
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    ProbAve = ProbAve/NumSamples;
    Sampled_Z = Z;
end
        
