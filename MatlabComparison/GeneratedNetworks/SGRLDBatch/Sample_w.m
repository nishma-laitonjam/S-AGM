function [UpdatedW, UpdatedSumWBar] = Sample_w(StepSize, alpha, Pi0, Pi, w, SumWBar, N, K, FullB, FullBTrainEdgeIndex, WMasksFull)

LogTrainEdgeProdAllK = sum(log(1 - w(FullBTrainEdgeIndex(:,2), :).*(w(FullBTrainEdgeIndex(:,3), :).*Pi)),2) + log(1 - Pi0);
FullB(FullBTrainEdgeIndex(:,1)) = LogTrainEdgeProdAllK - log(1 - exp(LogTrainEdgeProdAllK));
FullB = exp(FullB);
FullB = FullB.*WMasksFull;

FullWPi = w.*Pi;
WWpi = permute(FullWPi,[1, 3, 2]) .* permute(w, [3, 1, 2]); % N * N * K
CommonTerm = -permute(FullWPi, [3, 1, 2])./(1 - WWpi);
RowSum = squeeze(sum(CommonTerm.*full(FullB),2));

UpdatedW = ((1 - w)./SumWBar).*RowSum;  % Currently storing the common term when m = 0
UpdatedSumWBar = ((-w)./SumWBar).*RowSum; % Currently storing the common term when m = 1
WBar0 = w.*SumWBar;
WBar1 = (1 - w).*SumWBar;
xi = normrnd(0, sqrt(StepSize), [N, K, 2]);
UpdatedW = abs(WBar0 + (alpha - WBar0 + UpdatedW).*(StepSize/2) + sqrt(WBar0).*xi(:, :, 1));
UpdatedSumWBar = abs(WBar1 + (1 - WBar1 + UpdatedSumWBar).*(StepSize/2) + sqrt(WBar1).*xi(:, :, 2));
UpdatedSumWBar = UpdatedW + UpdatedSumWBar;
UpdatedW = UpdatedW./UpdatedSumWBar;
