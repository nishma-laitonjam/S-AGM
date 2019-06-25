function [B, ground_truth] = Generate(k, n, BackgroundProb, EdgeProb)

P0 = log(1 - BackgroundProb);
p = ones(1, k) .* EdgeProb;
Z = zeros(n, k);
non_overlapping = 0.4*n;
Z(1:non_overlapping,1) = 1;
Z((non_overlapping+1):2*non_overlapping,2) = 1;
Z((2*non_overlapping+1):n,1:2) = 1;

ZPZ = permute(Z.*p,[1,3,2]).* permute(Z, [3,1,2]);
logZPZ = log(1-ZPZ);
EdgeProb = 1 - exp(sum(logZPZ,3) + P0);

currAdj = zeros(n);
currAdj(rand(n)<EdgeProb)= 1;
B = triu(currAdj,1);
ground_truth = Z;
end
