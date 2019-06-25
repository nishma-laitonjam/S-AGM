cd ../GeneratedNetworks/GibbsAGM;
states = [0, 1, 2, 3, 4];
n = 100;
k = 2;
BackgroundProb = [0.00005, 0.0005, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
EdgeProb = 0.8;
for curr_BackgroundProb = BackgroundProb
    for curr_state = states
        rng(curr_state,'twister');
        
        [B, ground_truth] = Generate(k, n, curr_BackgroundProb, EdgeProb);
    
        Outfile = strcat("./Result/Gstate",num2str(curr_state),"n",...
            num2str(n),"k",num2str(k),"BackgroundProb",...
            num2str(curr_BackgroundProb),...
            "EdgeProb",num2str(EdgeProb),".mat");
        
        RunAGM(B, Outfile, k, ground_truth);
        
        clearvars -except states curr_state n k BackgroundProb...
            curr_BackgroundProb EdgeProb;
    end
end

