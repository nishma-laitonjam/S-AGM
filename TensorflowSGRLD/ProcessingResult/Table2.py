import numpy as np

Network = ["FASeed0L1000M1000K", "ReutersSeed0L1000M1000K", "ca-HepPhSeed0L1000M1000K"]
end_part = "burn_in2500samples2500test_ratio0.1"
commonpath = '../Results/'

table_for_time = np.zeros((3, 4))
table_for_auc = np.zeros((3, 4))

K_vec = [50, 100, 150, 200]
outer_count = 0
for K in K_vec:
    inner_count = 0
    for curr_network in Network:
        curr_npz = commonpath + curr_network + str(K) + end_part + '.npz'
        curr_results = np.load(curr_npz)
        TimeVector = curr_results['TimeVector']
        table_for_time[inner_count, outer_count] = TimeVector[-1] / 3600
        Avg_AUC = curr_results['AvgAUC']
        table_for_auc[inner_count, outer_count] = Avg_AUC
        inner_count = inner_count + 1
    outer_count = outer_count + 1
np.set_printoptions(precision=4)
print("Printing AUC")
print(table_for_auc)
print("Printing Time")
print(table_for_time)