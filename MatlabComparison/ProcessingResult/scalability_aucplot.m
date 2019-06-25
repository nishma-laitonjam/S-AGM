states = [0, 1, 2, 3, 4];
n = (1:10).*100;
k = 2;
avgauc = zeros(2, length(n));
iter = 0;
BackgroundProb = 0.00005;
EdgeProb = 0.8;
for curr_n = n
    iter = iter + 1;
    agm_auc = 0;
    sagm_auc = 0;
    for curr_state = states
        fileAGM = strcat("../GeneratedNetworks/GibbsAGM/Result/Gstate",...
            num2str(curr_state),"n",num2str(curr_n),"k",num2str(k),...
            "BackgroundProb",num2str(BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileAGM);
        agm_auc = agm_auc + AUCroc;
        clearvars -except curr_n n curr_state states k agm_auc...
            sagm_auc avgauc iter BackgroundProb EdgeProb;
        fileSAGM = strcat("../GeneratedNetworks/SGRLDBatch/Result/Gstate",...
            num2str(curr_state),"n", num2str(curr_n),"k",num2str(k),...
            "BackgroundProb",num2str(BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileSAGM);
        sagm_auc = sagm_auc + AUCroc;
        clearvars -except curr_n n curr_state states k agm_auc...
            sagm_auc avgauc iter BackgroundProb EdgeProb;
    end
    avgauc(1, iter) = agm_auc./length(states);
    avgauc(2, iter) = sagm_auc./length(states);
end
disp(avgauc);
graph = plot(avgauc(1,:), '-.r*');
hold on;
graph = plot(avgauc(2,:), '-.bo');
xlabel("Number of nodes in the network");
ylabel("AUC-ROC");
ylim([0.8 0.9]);
xlim([1 10]);
xticks(1:10);
xticklabels({'100', '200', '300', '400', '500', '600', '700', '800',...
    '900', '1000'});

set(graph,'LineWidth',1.5);
set(gca,'FontSize', 18);

legend('AGM', 'S-AGM', 'Location', 'northwest');
out_file = '../Plots/generatedgraph_scalability_aucplot';
saveas(gcf, out_file, 'epsc');
saveas(gcf, out_file, 'jpeg');
saveas(gcf, out_file, 'fig');



