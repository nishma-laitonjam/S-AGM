Model = ["AGM","AMMSB","SAGM"];
K = [5, 10, 15, 20];
Network = ["football_corrected", "nips_1-17", "Protein230"];
NumSamples = 5;
Mean_AUC = zeros(3, 12);
SD_AUC = zeros(3, 12);
outer_loop = 0;
for model = Model
    path = strcat('../SmallNetworks/Gibbs',model,"/Result/");
    outer_loop = outer_loop + 1;
    inner_loop = 0;
    for network = Network
        for k = K
            inner_loop = inner_loop + 1;
            AUCvec = zeros(1, NumSamples);
            for i=0:(NumSamples-1)
                load(strcat(path,network,num2str(i),'K',num2str(k),'.mat'));
                AUCvec(i+1) = AUCroc;
            end
            AUC = mean(AUCvec);
            Mean_AUC(outer_loop, inner_loop) = AUC;
            SD_AUC(outer_loop, inner_loop) = sqrt(mean((AUCvec - AUC).^2));
        end
    end
end
disp("Printing average of AUC");
disp(Mean_AUC);
disp("Printing standard deviation of AUC");
disp(SD_AUC);
