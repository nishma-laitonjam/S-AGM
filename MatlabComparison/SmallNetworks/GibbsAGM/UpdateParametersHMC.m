function [P] = UpdateParametersHMC(TestPairsZZ,TrainEdgesZZ,m,P,P0,a,b,TrainNumNonEdges)

% TrainEdgesZZ is for the 'Y1' and TestPairsZZ is for 'Y?'
NonEdgeZZ = [TrainEdgesZZ; TestPairsZZ];
NonEdgeTerm = transpose(m.*(m-1)/2-sum(NonEdgeZZ,1));

logpdf = @(r)Dist(r,a,b,NonEdgeTerm,TrainEdgesZZ,P0,TrainNumNonEdges);
startpoint = log(exp(-P)-1); % r


smp = hmcSampler(logpdf,startpoint, 'StepSize' , 0.01, 'NumSteps',10);
result = drawSamples(smp,'Burnin',0,'NumSamples',1);
finalR = transpose(result(size(result,1),:));
P = -log(1+exp(finalR));
