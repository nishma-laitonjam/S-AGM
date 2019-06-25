Network = ["FASeed0L1000M1000K", "ReutersSeed0L1000M1000K", "ca-HepPhSeed0L1000M1000K"];
end_part = "burn_in2500samples2500test_ratio0.1";
commonpath = '../Results/';
%---------------------------%
cd ../ProcessingResult/
pwd;
command = 'rm *.npy';
status = system(command);
count = 1;
for K = [50 100 150 200]
    command = 'rm *.npy';
    status = system(command);
    s = subplot(1,4,count);
    for network_iter = 1:length(Network)
        curr_network = Network(network_iter);
        npz = strcat(commonpath,curr_network,num2str(K),end_part,'.npz');
        command = char(strcat('unzip',{' '},npz));
        status = system(command);
        PP_path = './TestPPvector.npy';
        PP = readNPY(PP_path);
        graph = plot(PP,'-');
        set(graph,'LineWidth',1.5);
        set(gca,'FontSize', 18);
        ylim([0 40]);
        xlim([1 50]);
        set(gca, 'XTick', [1 10 20 30 40 50]);
        xlabel( 'After every 100 iterations' ) ;  ylabel( 'Perplexity' ) ;
        hold on;
        command = 'rm *.npy';
        status = system(command);
        title(strcat('K = ',num2str(K)));
    end
    l = legend("FreeAssoc","Reuters","ca-HepPh",'Location','northeast');
    l.FontSize = 18;
    count = count + 1;
end
out_file = strcat('../Plots/TestPP_Larger_Graph');
set(gcf, 'units','normalized','outerposition',[0 0 1.2 0.5]);
saveas(gcf,out_file,'fig');
saveas(gcf, out_file, 'jpeg');
clear all;
hold off;
%---------------------------%