Network = ["com-dblpSeed0L4096M32K2048burn_in25000samples25000test_ratio0.1"];
commonpath = '../Results/';
out_file = strcat('../Plots/com-dblp2048TestAUCPP_final');
%---------------------------%
cd ../ProcessingResult/
pwd;
command = 'rm *.npy';
status = system(command);
s = subplot(1,2,1);
for network_iter = 1:length(Network)
    curr_network = Network(network_iter);
    npz = strcat(commonpath,curr_network,'.npz');
    command = char(strcat('unzip',{' '},npz));
    status = system(command);
    %Plot the graphs
    AUC_path = './TestAUCvector.npy';
    auc = readNPY(AUC_path);
    graph = plot(auc,'-');
    set(graph,'LineWidth',1.5);
    set(gca,'FontSize', 18);
    ylim([0.5 1]);
    set(gca, 'XTick', [1 100 200 300 400 500]);
    xlabel( 'After every 100 iterations' ) ;  ylabel( 'AUC-ROC' ) ;
    hold on;
    %Remove the npy
    command = 'rm *.npy';
    status = system(command);
end
% l = legend("K = 512","K = 1024","K = 2048",'Location','northeast');
l.FontSize = 18;
%PP
s = subplot(1,2,2);
for network_iter = 1:length(Network)
    curr_network = Network(network_iter);
    npz = strcat(commonpath,curr_network,'.npz');
    command = char(strcat('unzip',{' '},npz));
    status = system(command);
    %Plot the graphs
    PP_path = './TestPPvector.npy';
    pp = readNPY(PP_path);
    graph = plot(pp,'-');
    set(graph,'LineWidth',1.5);
    set(gca,'FontSize', 18);
    ylim([0 150]);
    set(gca, 'XTick', [1 100 200 300 400 500]);
    xlabel( 'After every 100 iterations' ) ;  ylabel( 'Perplexity' ) ;
    hold on;
    %Remove the npy
    command = 'rm *.npy';
    status = system(command);
end
% l = legend("K = 512","K = 1024","K = 2048",'Location','northeast');
l.FontSize = 18;
set(gcf, 'units','normalized','outerposition',[0 0 1 0.5]);
saveas(gcf,out_file,'fig');
saveas(gcf, out_file, 'jpeg');
clear all;
hold off;
%---------------------------%