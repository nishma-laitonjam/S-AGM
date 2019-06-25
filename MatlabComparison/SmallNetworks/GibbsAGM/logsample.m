function [l,lsum] = logsample(logprob)
lsum = logprob(1);
for i = 2:length(logprob)
    lsum = logsum(lsum,logprob(i));
end
u = rand;
j = 1;
p = exp(logprob(1)-lsum);
while p < u
    j = j+1;
    p = p + exp(logprob(j)-lsum);
end
l = j;


