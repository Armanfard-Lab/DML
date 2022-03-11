import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size


def clustering_cost(x, y, u, p, u_means, landa):
    a = landa * torch.matmul(p, torch.sum(torch.pow(u - u_means, 2), dim=1))
    return [0.1*torch.matmul(p, torch.sum(torch.pow(x - y, 2), dim=1) ) + a, a]

def cluster_assignments(u, u_mean, batch_size, num_claster, m, network_list):
    p = torch.zeros([batch_size, num_claster]).cuda()
    new_count = 0
    for j in range(0, num_claster):
        if j in network_list:
            p[:, j] = torch.sum(torch.pow(u[new_count, :, :] - u_mean[j, :].unsqueeze(0).repeat(batch_size, 1), 2), dim=1)
            new_count +=1
        else:
            p[:, j] = torch.sum(torch.pow(u[-1, :, :] - u_mean[j, :].unsqueeze(0).repeat(batch_size, 1), 2), dim=1)
    p = torch.pow(p, -1 / (m - 1))
    sum1 = torch.sum(p, dim=1)

    p = torch.div(p, sum1.unsqueeze(1).repeat(1, num_claster))
    # print(p[1,:])
    return p





























