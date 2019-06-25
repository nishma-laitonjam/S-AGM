state = 0;
Model = ["AGM", "SAGM"];
Network = ["football_corrected", "nips_1-17", "Protein230"];
K = [5,10,15,20];
commonpath = '../SmallNetworks/Gibbs';
%---------------------------%
iter = 1;
for model_iter = 1:length(Model)
    curr_model = Model(model_iter);
    for network_iter = 1:length(Network)
        curr_network = Network(network_iter);
        s = subplot(length(Model),length(Network),iter);
        iter = iter + 1;
        for k_iter = 1:length(K)
            curr_k = K(k_iter);
            path = strcat(commonpath,curr_model,'/Result/',curr_network,num2str(state),'K',num2str(curr_k),'.mat');
            load(path);
            AveEdgeProb = Total_Probability_vs_num_shared_community(1,:)./(Total_Probability_vs_num_shared_community(2,:));
            graph = plot(0:length(AveEdgeProb) - 1, AveEdgeProb(1:end));
            set(graph,'LineWidth',1.5);
            hold on;
        end
        set(gca,'FontSize', 18);
        ylabel( 'Edge probability' ) ;  xlabel( 'Number of shared communities' ) ;
        if strcmp(curr_model,'SAGM')
            if strcmp(curr_network,"football_corrected")
                t = title(sprintf('S-AGM with Football'));
            elseif strcmp(curr_network,"nips_1-17")
                t = title(sprintf('S-AGM with NIPS234'));
            elseif strcmp(curr_network,"Protein230")
                t = title(sprintf('S-AGM with Protein230'));
            end  
        else
            if strcmp(curr_network,"football_corrected")
                t = title(sprintf('AGM with Football'));
            elseif strcmp(curr_network,"nips_1-17")
                t = title(sprintf('AGM with NIPS234'));
            elseif strcmp(curr_network,"Protein230")
                t = title(sprintf('AGM with Protein230'));
            end  
        end
        set(t,'Interpreter', 'none')
        l = legend("K = 5","K = 10","K = 15","K = 20",'Location','southeast');
        l.FontSize = 10;
    end
end
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
out_file = strcat('../Plots/State',num2str(state),'PluralisticHomophily');
saveas(gcf, out_file, 'epsc');
saveas(gcf, out_file, 'jpeg');
saveas(gcf, out_file, 'fig');
%---------------------------%