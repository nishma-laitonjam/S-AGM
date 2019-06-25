cd ../GeneratedNetworks/GibbsAGM;
states = [0, 1, 2, 3, 4];
n = 100;
k = 2;
BackgroundProb = 0.00005;
EdgeProb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
for curr_EdgeProb = EdgeProb
    for curr_state = states
        rng(curr_state,'twister');
        [B, ground_truth] = Generate(k, n, BackgroundProb, ...
            curr_EdgeProb);
        
        Outfile = strcat("./Result/Gstate",num2str(curr_state),"n",...
            num2str(n),"k",num2str(k),"BackgroundProb",...
            num2str(BackgroundProb),...
            "EdgeProb",num2str(curr_EdgeProb),".mat");
        
        RunAGM(B, Outfile, k, ground_truth);
        clearvars -except states curr_state n k BackgroundProb...
            EdgeProb curr_EdgeProb;
    end
end


