function [n_i] = find_n_i(Z,N,K)
n_i = zeros(N,K);
for k = 1:K
    n_i(:, k) = sum(Z==k,2);
end
% for i=1:N
%     for k = 1:K
%         n_i(i,k) = sum(Z(i,:)==k);
%     end
% end