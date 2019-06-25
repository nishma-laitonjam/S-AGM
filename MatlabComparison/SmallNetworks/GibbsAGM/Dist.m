function [lpdf,glpdf] = Dist(r,a,b,NonEdgeTerm,TrainEdgesZZ,P0,TrainNumNonEdges)

Rho = 1./(1+exp(-r));
P = log(1-Rho);
ZZp = transpose(P).*TrainEdgesZZ;
ZPZ = sum(ZZp,2)+P0;

Edge = sum(log(1-exp(ZPZ)));

NonEdge = sum(NonEdgeTerm.*P)+P0*TrainNumNonEdges;
lpdf = Edge + NonEdge + sum(a.*r-(a+b).*log(1+exp(r)));

ZZRho = transpose(Rho).*TrainEdgesZZ;
Edge = transpose(sum(exp(ZPZ)./(1-exp(ZPZ)).*ZZRho,1));

NonEdge = -NonEdgeTerm.*Rho;
glpdf = Edge + NonEdge -(a+b).*Rho+a;