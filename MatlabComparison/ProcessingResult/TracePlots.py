# Wrote by Weipeng Huang
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("white")
sns.set_palette('Set1')
sns.set_style("ticks")

import os
import scipy.io as spio

startIter = 10;
endIter = 3000;
state = 0;
Model = ["AGM", "AMMSB", "SAGM"];
Network = ["football_corrected", "nips_1-17", "Protein230"];
K = [5,10,15,20];
commonpath = '../SmallNetworks/'

nwd = {
    "football_corrected": 'Football',
    "nips_1-17": 'NIPS234',
    "Protein230": 'Protein230'
}

loc = (.51, .0)
fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(13, 8))
for i, network in enumerate(Network):
    for j, k in enumerate(K):
        for model in Model:
            path = os.path.join(commonpath, '{}{}'.format('Gibbs', str(model)), 'Result',
                                '{}{}{}{}'.format(network, str(state),
                                                  'K', str(k), '.mat'))
            f = spio.loadmat(path)
            data = f['LLvector']

            axes[i, j].plot(data, linestyle='-', linewidth=1.5)
            if i == 0:
                axes[i, j].set_title('K = {}'.format(k), fontsize=20)
            if j == 0:
                axes[i, j].yaxis.set_tick_params(labelsize=16)
                axes[i, j].set_ylabel('{} \n Log Likelihood'.format(nwd[network]), fontsize=18)
            if i == 2:
                axes[i, j].xaxis.set_tick_params(labelsize=16)

plt.ylim((-4000, 0))
plt.xlim((1, 5000))
# plt.xticks([1, 1000, 2000, 3000, 4000, 5000])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Iterations', fontsize=18)
plt.figlegend(labels=["AGM", "a-MMSB", "S-AGM"], loc=loc, fontsize=16, ncol=3)
# plt.tight_layout()
plt.savefig('../Plots/State0TracePlotTrainLL.eps', dpi=900)

loc = (.51, -.0)
fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(13, 8))
for i, network in enumerate(Network):
    for j, k in enumerate(K):
        for model in Model:
            path = os.path.join(commonpath, '{}{}'.format('Gibbs', str(model)), 'Result',
                                '{}{}{}{}'.format(network, str(state),
                                                  'K', str(k), '.mat'))
            f = spio.loadmat(path)
            data = f['PerplexityVector']

            axes[i, j].plot(data, linestyle='-', linewidth=1.5)
            if i == 0:
                axes[i, j].set_title('K = {}'.format(k), fontsize=20)
            if j == 0:
                axes[i, j].set_ylabel('{} \n Perplexity'.format(nwd[network]), fontsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=16)
            if i == 2:
                axes[i, j].xaxis.set_tick_params(labelsize=16)

plt.ylim((1, 1.5))
plt.xticks([1, 10, 20, 30, 40, 50])

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('After every 100 iterations', fontsize=18)

plt.figlegend(labels=["AGM", "a-MMSB", "S-AGM"], loc=loc, fontsize=16, ncol=3)
# plt.tight_layout()
plt.savefig('../Plots/State0TracePlotPerplexity.eps', dpi=900)
