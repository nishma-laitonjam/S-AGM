Network = ["ca-HepPhSeed0L1000M1000K50test_ratio0.1Time", "ca-HepPhSeed0L500M500K50test_ratio0.1Time", "ca-HepPhSeed0L100M100K50test_ratio0.1Time", "ca-HepPhSeed0L5M5K50test_ratio0.1Time"];
commonpath = '../Results/';
out_file = strcat('../Plots/caHepPhTestAUCPP_final');
%---------------------------%
cd ../ProcessingResult/
pwd;
command = 'rm *.npy';
status = system(command);
%AUC
s = subplot(1,2,1);
for network_iter = 1:length(Network)
    curr_network = Network(network_iter);
    npz = strcat(commonpath,curr_network,'.npz');
    command = char(strcat('unzip',{' '},npz));
    status = system(command);
    %Plot the graphs
    AUC_path = './TestAUCvector.npy';
    auc = readNPY(AUC_path);
    x = 1:length(auc);
    x_sec = x.*30;
    x_min = x_sec./60;
    graph = plot(x_min, auc,'-');
    set(graph,'LineWidth',1.5);
    set(gca,'FontSize', 18);
    ylim([0.5 1]);
    set(gca, 'XTick', [0.5 10 20 30 40 50 60]);
    xlim([x_min(1) x_min(end)]);
    xlabel( 'Time (minutes)' ) ;  ylabel( 'AUC-ROC' ) ;
    hold on;
    command = 'rm *.npy';
    status = system(command);
end
l = legend("Mini-batch = 1000","Mini-batch = 500","Mini-batch = 100","Mini-batch = 5",'Location','southeast');
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
    x = 1:length(auc);
    x_sec = x.*30;
    x_min = x_sec./60;
    xlim([x_min(1) x_min(end)]);
    graph = plot(x_min, pp,'-');
    set(graph,'LineWidth',1.5);
    set(gca,'FontSize', 18);
    ylim([0 50]);
    set(gca, 'XTick', [0.5 10 20 30 40 50 60]);
    xlabel( 'Time (minutes)' ) ;  ylabel( 'Perplexity' ) ;
    hold on;
    command = 'rm *.npy';
    status = system(command);
end
l = legend("Mini-batch = 1000","Mini-batch = 500","Mini-batch = 100","Mini-batch = 5",'Location','northeast');
l.FontSize = 18;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
saveas(gcf, out_file, 'fig');
saveas(gcf, out_file, 'jpeg');
saveas(gcf, out_file, 'epsc');
clear all;
hold off;
%---------------------------%