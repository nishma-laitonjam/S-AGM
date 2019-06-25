function r = drchrnd(a)
% take a sample from a dirichlet distribution
r = gamrnd(a,1);
r = r ./ sum(r,2);