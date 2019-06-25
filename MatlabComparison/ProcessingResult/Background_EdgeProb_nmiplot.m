states = [0, 1, 2, 3, 4];
threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
n = 100;
k = 2;
BackgroundProb = [0.00005, 0.0005, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
EdgeProb = 0.8;
Avg_sagm_nmi = zeros(length(threshold), length(BackgroundProb));
Avg_agm_nmi = zeros(1, length(BackgroundProb));
iter = 0;
for curr_BackgroundProb = BackgroundProb
    iter = iter+1;
    % S-AGM
    curr_sagm_nmi = zeros(1, length(threshold));
    for curr_state = states
        fileSAGM = strcat("../GeneratedNetworks/SGRLDBatch/Result/Gstate",...
            num2str(curr_state),"n",num2str(n),"k",num2str(k),...
            "BackgroundProb",num2str(curr_BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileSAGM);
        curr_sagm_nmi = curr_sagm_nmi + nmi;
    end
    Avg_sagm_nmi(:, iter) = curr_sagm_nmi./length(states);
    curr_agm_nmi = 0;
    for curr_state = states
        fileAGM = strcat("../GeneratedNetworks/GibbsAGM/Result/Gstate",...
            num2str(curr_state),"n",num2str(n),"k",num2str(k),...
            "BackgroundProb",num2str(curr_BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileAGM);
        curr_agm_nmi = curr_agm_nmi + nmi;
    end
    Avg_agm_nmi(iter) = curr_agm_nmi./length(states);
end
graph = plot(Avg_agm_nmi, '-.r*');
hold on;

% S-AGM
for i = 6
    graph = plot(Avg_sagm_nmi(i, :), '-.bo');
end
xlim([1, 9]);
xticks([1, 2, 3, 4, 5, 6, 7, 8, 9]);
xlabel("Background Probability");
ylabel("NMI");
xticklabels({'5e-05', '5e-04', '5e-03', '5e-02', '0.1', '0.2', '0.3', ...
    '0.4', '0.5'});

set(graph,'LineWidth',1.5);
set(gca,'FontSize', 18);

legend('AGM', 'S-AGM with 0.5', 'Location', 'SouthWest');
out_file = '../Plots/generatedgraph_Background_EdgeProb_nmiplot';
saveas(gcf, out_file, 'epsc');
saveas(gcf, out_file, 'jpeg');
saveas(gcf, out_file, 'fig');

