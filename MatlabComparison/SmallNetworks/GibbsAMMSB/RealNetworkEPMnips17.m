K_values = [5, 10, 15, 20];
NumRandom = 5;

cd ../SmallNetworks/GibbsAMMSB;

file = "../../Datasets/nips_1-17.mat";
Data = load(file);
DocAuthor = Data.docs_authors;
B = zeros(size(DocAuthor,2));
for i=1:size(DocAuthor,2)
    B(i,:) = sum( DocAuthor(DocAuthor(:,i)>0,:),1)>0;
end
Coauthor_times = sum(B,1);
[~,idex]=sort(Coauthor_times,'descend');
B = B(idex(1:234),idex(1:234));
B = triu(B,1);

clearvars -except B K_values NumRandom;
for K_iter = 1:length(K_values)
    for state = 0:(NumRandom-1)
        rng(state,'twister');
        K = K_values(K_iter);
        Outfile = strcat("./Result/nips_1-17",num2str(state),"K",num2str(K),".mat");
        RunAMMSB(B, Outfile, K);
        clearvars -except K_values K_iter NumRandom state B;
    end
end

clear all;
%-----------------------------------------%