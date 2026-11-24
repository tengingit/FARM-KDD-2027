import torch


def compute_all_priors(Y_train, num_per_dim):

    num_training, num_dim = Y_train.shape
    max_c = max(num_per_dim)

    alpha = torch.zeros(num_dim, max_c)
    weight_gamma = torch.zeros(num_dim, max_c, num_dim)

    cond_prob = torch.zeros(num_dim, max_c, num_dim, max_c)

    for j1 in range(num_dim):
        K1 = num_per_dim[j1]

        counts = torch.bincount(Y_train[:, j1], minlength=K1).float()

        alpha[j1, :K1] = 1.0 / (torch.sqrt(counts) + 1.0)

        for c1 in range(K1):
            mask = (Y_train[:, j1] == c1)
            n_c1 = mask.sum().item()

            if n_c1 == 0: continue

            for j2 in range(num_dim):
                if j2 == j1:

                    cond_prob[j1, c1, j2, c1] = 1.0
                    continue

                labels_in_j2 = Y_train[mask, j2]
                K2 = num_per_dim[j2]

                dist_j2 = torch.bincount(labels_in_j2, minlength=K2).float()
                cond_prob[j1, c1, j2, :K2] = dist_j2 / n_c1

                unique_labels_j2 = torch.unique(labels_in_j2)

                weight_gamma[j1, c1, j2] = 1.0 - (len(unique_labels_j2) / K2)

    return alpha, weight_gamma, cond_prob
