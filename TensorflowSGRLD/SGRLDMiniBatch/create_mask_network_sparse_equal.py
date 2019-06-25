import numpy as np
import math


def create_mask_network_sparse_equal(B, num_test_pairs, num_train_pairs):

    n = B.shape[1]

    # Sampling half of test set as edge
    num_test_pairs_edge = int(num_test_pairs / 2)
    num_train_pairs_edge = int(num_train_pairs / 2)
    edge_pairs = num_test_pairs_edge + num_train_pairs_edge
    random_idx = np.random.randint(1, B.nnz, 3 * edge_pairs)

    idx = np.stack((B.nonzero()[0][random_idx], B.nonzero()[1][random_idx]))
    idx = np.unique(idx, axis=1)

    idx_test_edges = idx[:, 0:num_test_pairs_edge]
    idx_train_edges = idx[:, num_test_pairs_edge:edge_pairs]


    # Sampling other half of test set as non edge
    num_test_pairs_non_edge = int(num_test_pairs / 2)
    num_train_pairs_non_edge = int(num_train_pairs / 2)
    non_edge_pairs = num_test_pairs_non_edge + num_train_pairs_non_edge
    random_rows_idx = np.random.choice(n, 3*non_edge_pairs, replace=True)
    random_cols_idx = np.random.choice(n, 3*non_edge_pairs, replace=True)
    idx = np.stack((random_rows_idx, random_cols_idx))
    idx = np.unique(idx, axis=1)
    is_test = np.zeros((1, len(idx[0])), dtype=bool)

    count = 0
    for i in range(0, len(idx[0])):  # 0 to (len - 1)
        if idx[0, i] < idx[1, i]:
            if count >= non_edge_pairs:
                break
            if B[idx[0, i], idx[1, i]] == 0:
                count = count + 1
                is_test[0, i] = True


    idx_non_edges = idx[:, np.where(is_test[0, :])[0]]
    idx_test_non_edges = idx_non_edges[:, 0:num_test_pairs_non_edge]
    idx_train_non_edges = idx_non_edges[:, num_test_pairs_non_edge:non_edge_pairs]
    idx_test = np.hstack((idx_test_edges, idx_test_non_edges))
    idx_train = np.hstack((idx_train_edges, idx_train_non_edges))

    # Random sequence of test pairs
    # rand_T_idx = np.random.permutation(T)
    # idx_test = idx_test[:, rand_T_idx]

    return [idx_test, idx_train]

