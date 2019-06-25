function [W] = Sample_W(alpha, m, N)
W = betarnd(alpha + m, N + 1 - m);