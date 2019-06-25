function [P] = UpdateParametersHMC(CountTrainNonEdgeEachCluster,TrainEdgeZZ,P,P0,eta_0,eta_1,NumTrainNonEdges)

logpdf = @(r)Dist(r,eta_0,eta_1,CountTrainNonEdgeEachCluster,TrainEdgeZZ,P0,NumTrainNonEdges);

startpoint = log(exp(-P)-1); % r


smp = hmcSampler(logpdf,startpoint, 'StepSize' , 0.01, 'NumSteps',10);

result = drawSamples(smp,'Burnin',0,'NumSamples',1);
finalR = result(size(result,1),:);
P = -log(1+exp(finalR));% row vector
