function [UpdatedPi, UpdatedSumPiBar] = Sample_pi(TrainEdges, TrainNonEdges, StepSize, eta_0, eta_1, Pi0, Pi, SumPiBar, w, K)

xi = normrnd(0, sqrt(StepSize), [2,K]);% Sample the gaussian noise

%Calculating H (Train Edge)
WWTrainEdge = w(TrainEdges(:,1),:).*w(TrainEdges(:,2),:);
LogTermTrainEdge = log(1-WWTrainEdge.*Pi);
SumLogTermTrainEdge = sum(LogTermTrainEdge,2)+log(1-Pi0);
TrainEdgeH = (WWTrainEdge.*exp(SumLogTermTrainEdge-LogTermTrainEdge))./(1-exp(SumLogTermTrainEdge));

%Calculating H (Train Non Edge)
WWTrainNonEdge = w(TrainNonEdges(:,1),:).*w(TrainNonEdges(:,2),:);
TrainNonEdgeH = -WWTrainNonEdge./(1-WWTrainNonEdge.*Pi);

%Calculating G
CommonTerm = [1-Pi;-Pi]./SumPiBar; %CommonTerm = [-Pi;(1-Pi)]./SumPiBar;
SumTrainPairG = CommonTerm.*(sum(TrainEdgeH,1)+sum(TrainNonEdgeH,1));

PiBar = [Pi;1-Pi].*SumPiBar; %PiBar = [1-Pi;Pi].*SumPiBar;

%Sample New Value 
UpdatedPiBar = abs(PiBar+sqrt(PiBar).*xi+([eta_0;eta_1]-PiBar+SumTrainPairG).*(StepSize/2));

UpdatedSumPiBar = sum(UpdatedPiBar,1);
UpdatedPi = UpdatedPiBar(1,:)./UpdatedSumPiBar; %UpdatedPi = UpdatedPiBar(2,:)./UpdatedSumPiBar;



