function m_k = find_m_k(Z, K, B, B_NonEdge)
m_k = zeros(K,2);
Index = find(triu(Z,1)== triu(Z',1));
for k = 1:K
    m_k(k,1) = sum((Z(Index) == k).*B(Index));
    m_k(k,2) = sum((Z(Index) == k).*B_NonEdge(Index));
end