function [Sampled_Z, ProbAve, Sampled_pi, Avg_pi, Avg_beta, Sampled_n_i, AUCvector, PerplexityVector, LLvector, TimeVector] = aMMSB_NC_mcmc(test_link, NumSamples, BurnIn, TrainEdges, TrainNonEdges, TestPairs, Z, n_i, m_k, pi, beta, a, b, alpha, epsilon, T)

K = size(pi,2);
N = size(pi,1);
NumTestPairs = size(test_link,1);
ProbAve = zeros(NumTestPairs,1);
Total_pi = zeros(N,K);
Total_beta = zeros(K,1);
TotalIter = NumSamples + BurnIn;

AUCvector = [];
PerplexityVector = [];
LLvector = [];
TimeVector = [];
if TotalIter > 0
    for iter=1:BurnIn
        iter_tic = tic;
        
        % Sample Z
        [Z, n_i, m_k] = sample_Z_parallel(TrainEdges, TrainNonEdges, Z, n_i, m_k, K, pi, beta, epsilon);
        
        % Sample pi
        pi = drchrnd(n_i + alpha);
        
        % Sample Beta
        beta = betarnd(m_k(:,1) + a, m_k(:,2) + b);
        
        % Compute Log Likelihood of training data
        LLTrainEdge = FindLogP(TrainEdges, pi, beta, epsilon);
        LLTrainNonEdge = log(1- exp(FindLogP(TrainNonEdges, pi, beta, epsilon)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];
        
        if mod(iter,T)==0
            disp(iter);
            test_logP = FindLogP(TestPairs, pi, beta, epsilon);
            [~,~,~,AUCroc] = perfcurve(test_link,test_logP,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(test_logP(test_link==1),1)+sum(log(1-exp(test_logP(test_link==0))),1);
            Perp = exp(-(SumTestLL)/NumTestPairs);
            PerplexityVector = [PerplexityVector; Perp];
        end
        
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    
    for iter=(BurnIn+1):(BurnIn+NumSamples)
        iter_tic = tic;
        
        % Sample Z
        [Z, n_i, m_k] = sample_Z_parallel(TrainEdges, TrainNonEdges, Z, n_i, m_k, K, pi, beta, epsilon);
        
        % Sample pi
        pi = drchrnd(n_i + alpha);
        
        % Sample Beta
        beta = betarnd(m_k(:,1) + a, m_k(:,2) + b);
        
        % Compute Log Likelihood of training data
        LLTrainEdge = FindLogP(TrainEdges, pi, beta, epsilon);
        LLTrainNonEdge = log(1- exp(FindLogP(TrainNonEdges, pi, beta, epsilon)));
        LLvector = [LLvector; sum(LLTrainNonEdge,1) + sum(LLTrainEdge,1)];
        
        % Output ProbAve
        test_logP = FindLogP(TestPairs, pi, beta, epsilon);
        if mod(iter,T)==0
            disp(iter);
            [~,~,~,AUCroc] = perfcurve(test_link,test_logP,1);
            disp(AUCroc);
            AUCvector = [AUCvector ; AUCroc];
            % Perplexity computations
            SumTestLL = sum(test_logP(test_link==1),1)+sum(log(1-exp(test_logP(test_link==0))),1);
            Perp = exp(-(SumTestLL)/NumTestPairs);
            PerplexityVector = [PerplexityVector; Perp];
        end
        ProbAve = ProbAve + test_logP;
        Total_pi = Total_pi + pi;
        Total_beta = Total_beta + beta;
        iter_toc = toc(iter_tic);
        TimeVector = [TimeVector ; iter_toc];
    end
    ProbAve = ProbAve/NumSamples;
    Sampled_Z = Z;
    Sampled_pi = pi;
    Avg_pi = Total_pi/NumSamples;
    Avg_beta = Total_beta/NumSamples;
    Sampled_n_i = n_i;
end

        
