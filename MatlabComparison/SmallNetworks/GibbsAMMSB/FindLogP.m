function LogP = FindLogP(NodesPairs, pi, beta, epsilon)

i = NodesPairs(:,1);
j = NodesPairs(:,2);
PiPi = pi(i,:).*pi(j,:);
P = sum(beta'.*PiPi,2) + epsilon*(1-sum(PiPi,2));
LogP = log(P);