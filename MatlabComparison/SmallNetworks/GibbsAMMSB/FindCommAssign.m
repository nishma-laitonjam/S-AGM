function CommAssign = FindCommAssign(Sampled_Z, K)
N = size(Sampled_Z,1);
CommAssign = zeros(N, K);
for i = 1:N
    for k = 1:K
        if sum(Sampled_Z(i,:)==k)>1
            CommAssign(i,k) = 1;
        end
    end
end