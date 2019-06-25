function [Avg_TestPairProb, Avg_w, Avg_pi, Sampled_Z, Total_Probability_vs_num_shared_community, AUCvector, PerplexityVector, LLvector, TimeVector]= SAGMmcmc(TrainEdges,TrainNonEdges,TrainEdgeZZ,eta_0,eta_1,alpha,alpha_shape,alpha_rate,P0,pi,P,w,NumSamples,NumBurnIn,CountTestEachNode,NumTrainNonEdges,TestPairs,test_link,T,GroundTruth)
if ~exist('GroundTruth','var')
    GroundTruth = []; 
end
TotalIter = NumSamples + NumBurnIn;

K = size(pi,2);
N = size(w,1);
disp(K);
disp(N);

Total_Probability_vs_num_shared_community = zeros(2, K+1);

TotalTestPairProb = zeros(size(TestPairs,1),1);
Total_w = zeros(N,K);
Total_pi = zeros(1,K);

AUCvector = [];
PerplexityVector = [];
LLvector = [];
TimeVector = [];
if TotalIter > 0
    for iter = 1:NumBurnIn
        iter_tic = tic;
        % Sample Z % calculate many parts for HMC in Z
        [z_ik, CountTrainNonEdgeEachCluster, TrainEdgeZZ, Probability_vs_num_shared_community] = Sample_Z(TrainEdges,TrainNonEdges,N,K,w,P,P0,TrainEdgeZZ);

        % Sample w
        w = Sample_w(N,alpha,z_ik,CountTestEachNode);

        % Sample pi
        P = Sample_pi(eta_0,eta_1,P,P0,CountTrainNonEdgeEachCluster, TrainEdgeZZ,NumTrainNonEdges);
        pi = 1-exp(P);
        
        % Sample alpha
        alpha = Sample_alpha(w,N,alpha_shape,alpha_rate);
        
        % Compute Log Likelihood of training data
        LLTrainNonEdge = P0 + sum(log(1-w(TrainNonEdges(:,1),:).*w(TrainNonEdges(:,2),:).*pi),2);
        LLTrainEdge = log(1-exp(P0 + sum(log(1-w(TrainEdges(:,1),:).*w(TrainEdges(:,2),:).*pi),2)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];

        % Print current iteration
        if mod(iter,T)==0
            fprintf("Number of iterations = %d\n",iter);
            disp(alpha) % Print the communities
            % calculate AUC-ROC without sampling Z
            StartNode_prob = w(TestPairs(:,1),:);
            EndNode_prob = w(TestPairs(:,2),:);
            LogSumTest = P0+sum(log(1-StartNode_prob.*EndNode_prob.*pi),2);
            CurrTestPairProb = 1-exp(LogSumTest);
            [~,~,~,AUCroc] = perfcurve(test_link,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(LogSumTest(test_link==0),1)+sum(log(1-exp(LogSumTest(test_link==1))),1);
            Perp = exp(-(SumTestLL)/size(test_link,1));
            PerplexityVector = [PerplexityVector; Perp];
        end
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    for iter = (NumBurnIn+1):(NumBurnIn+NumSamples)
        iter_tic = tic;
        % Sample Z % calculate many parts for HMC in Z
        [z_ik, CountTrainNonEdgeEachCluster,TrainEdgeZZ, Probability_vs_num_shared_community] = Sample_Z(TrainEdges,TrainNonEdges,N,K,w,P,P0,TrainEdgeZZ);

        % Sample w
        w = Sample_w(N,alpha,z_ik,CountTestEachNode);

        % Sample pi
        P = Sample_pi(eta_0,eta_1,P,P0,CountTrainNonEdgeEachCluster,TrainEdgeZZ,NumTrainNonEdges);
        pi = 1-exp(P);
        
        % Sample alpha
        alpha = Sample_alpha(w,N,alpha_shape,alpha_rate);
        
        % Compute Log Likelihood of training data
        LLTrainNonEdge = P0 + sum(log(1-w(TrainNonEdges(:,1),:).*w(TrainNonEdges(:,2),:).*pi),2);
        LLTrainEdge = log(1-exp(P0 + sum(log(1-w(TrainEdges(:,1),:).*w(TrainEdges(:,2),:).*pi),2)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];

        Total_Probability_vs_num_shared_community = Total_Probability_vs_num_shared_community + Probability_vs_num_shared_community;
        
        % calculate AUC-ROC without sampling Z
        StartNode_prob = w(TestPairs(:,1),:);
        EndNode_prob = w(TestPairs(:,2),:);
        LogSumTest = P0+sum(log(1-StartNode_prob.*EndNode_prob.*pi),2);
        CurrTestPairProb = 1-exp(LogSumTest);
        % CurrTestPairProb = 1-exp(P0+sum(log(1-StartNode_prob.*EndNode_prob.*pi),2));
        % Print current iteration
        if mod(iter,T)==0
            fprintf("Number of iterations = %d\n",iter);
            disp(alpha); % Print the communities
            [~,~,~,AUCroc] = perfcurve(test_link,CurrTestPairProb,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            fprintf("Average Edge Probability");
            disp(Total_Probability_vs_num_shared_community(1,:)./(Total_Probability_vs_num_shared_community(2,:)));
            % Perplexity computations
            SumTestLL = sum(LogSumTest(test_link==0),1)+sum(log(1-exp(LogSumTest(test_link==1))),1);
            Perp = exp(-(SumTestLL)/size(test_link,1));
            PerplexityVector = [PerplexityVector; Perp];
        end
        
        % output data
        TotalTestPairProb = TotalTestPairProb + CurrTestPairProb;
        Total_w = Total_w + w;
        Total_pi = Total_pi + pi;
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    Avg_TestPairProb = TotalTestPairProb/NumSamples;
    Avg_pi = Total_pi/NumSamples;
    Avg_w = Total_w/NumSamples;
    Sampled_Z = z_ik;
end
