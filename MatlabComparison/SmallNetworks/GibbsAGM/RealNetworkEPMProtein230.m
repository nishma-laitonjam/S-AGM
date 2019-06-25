K_values = [5, 10, 15, 20];
NumRandom = 5;
cd ../SmallNetworks/GibbsAGM;

file = "../../Datasets/Protein230.mat";
Data = load(file);
B = Data.B;
B = triu(B,1);

clearvars -except B K_values NumRandom;
for K_iter = 1:length(K_values)
    for state = 0:(NumRandom-1)
        rng(state,'twister');
        K = K_values(K_iter);
        
        Outfile = strcat("./Result/Protein230",num2str(state),"K",num2str(K),".mat");
        RunAGM(B, Outfile, K);
        
        clearvars -except K_values K_iter NumRandom state B;
    end
end

clear all
%-----------------------------------------%
