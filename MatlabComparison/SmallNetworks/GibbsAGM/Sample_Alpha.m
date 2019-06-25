function [alpha] = Sample_Alpha(gamma0, gamma1, K, W)
alpha = gamrnd(K + gamma0, 1/(gamma1-sum(log(W),2)));