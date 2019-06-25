states = [0, 1, 2, 3, 4];
n = (1:10).*100;
k = 2;
avgtime = zeros(2, length(n));
iter = 0;
BackgroundProb = 0.00005;
EdgeProb = 0.8;
for curr_n = n
    iter = iter + 1;
    agm_time = 0;
    sagm_time = 0;
    for curr_state = states
        fileAGM = strcat("../GeneratedNetworks/GibbsAGM/Result/Gstate",...
            num2str(curr_state),"n",num2str(curr_n),"k",num2str(k),...
            "BackgroundProb",num2str(BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileAGM);
        agm_time = agm_time + time;
        clearvars -except curr_n n curr_state states k agm_time...
            sagm_time avgtime iter BackgroundProb EdgeProb;
        fileSAGM = strcat("../GeneratedNetworks/SGRLDBatch/Result/Gstate",...
            num2str(curr_state),"n", num2str(curr_n),"k",num2str(k),...
            "BackgroundProb",num2str(BackgroundProb),"EdgeProb",...
            num2str(EdgeProb),".mat");
        load(fileSAGM);
        sagm_time = sagm_time + stop_time;
        clearvars -except curr_n n curr_state states k agm_time...
            sagm_time avgtime iter BackgroundProb EdgeProb;
    end
    avgtime(1, iter) = agm_time./length(states);
    avgtime(2, iter) = sagm_time./length(states);
end
disp(avgtime);
graph = plot(avgtime(1,:), '-.r*');
hold on;
graph = plot(avgtime(2,:), '-.bo');
xlabel("Number of nodes in the network");
ylabel("Time (sec)");
xlim([1 10]);
xticks(1:10);
xticklabels({'100', '200', '300', '400', '500', '600', '700', '800',...
    '900', '1000'});

set(graph,'LineWidth',1.5);
set(gca,'FontSize', 18);

legend('AGM', 'S-AGM', 'Location', 'northwest');
out_file = '../Plots/generatedgraph_scalability_timeplot';
saveas(gcf, out_file, 'epsc');
saveas(gcf, out_file, 'jpeg');
saveas(gcf, out_file, 'fig');
