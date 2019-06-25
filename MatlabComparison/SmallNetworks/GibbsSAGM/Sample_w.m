function [w] = Sample_w(N,alpha,z_ik,CountTestEachNode)
a = alpha; % alpha is row vector
b = 1;% change according to the model you choose IBP or BP
% b = 0.1; % Modified for fixed alpha

w = betarnd(a + z_ik, b + N - 1 - CountTestEachNode - z_ik);% self edge and test edge