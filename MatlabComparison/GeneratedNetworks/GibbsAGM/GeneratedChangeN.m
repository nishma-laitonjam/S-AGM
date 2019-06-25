cd ../GeneratedNetworks/GibbsAGM;
states = [0, 1, 2, 3, 4];
n = (1:10) .* 100;
k = 2;
BackgroundProb = 0.00005;
EdgeProb = 0.8;
for curr_n = n
    for curr_state = states
        rng(curr_state,'twister');
        [B, ground_truth] = Generate(k, curr_n, BackgroundProb, EdgeProb);
    
        Outfile = strcat("./Result/Gstate",num2str(curr_state),"n",...
            num2str(curr_n),"k",num2str(k),"BackgroundProb",...
            num2str(BackgroundProb),...
            "EdgeProb",num2str(EdgeProb),".mat");
            
        RunAGM(B, Outfile, k, ground_truth);
        clearvars -except states curr_state n curr_n k BackgroundProb EdgeProb;
    end
end