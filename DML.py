import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils import acc, cluster_assignments, clustering_cost
from sklearn.metrics import normalized_mutual_info_score


nmi = normalized_mutual_info_score


class DML(nn.Module):
    def __init__(self,AE_net, train_loader, batch_size, optimizer, cluster_centers, num_clusters, m, network_list, dataset_size, landa):
        super(DML, self).__init__()
        self.AE_net = AE_net
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.u_mean = cluster_centers
        self.num_claster = num_clusters
        self.m= m
        self.network_list = network_list
        self.dataset_size = dataset_size
        self.landa = landa

    def update_cluster_centers(self):

        true_label = np.zeros(self.dataset_size)
        pred_label = np.zeros(self.dataset_size)
        ii = 0
        u_cent = torch.zeros([self.num_claster, self.batch_size, 10]).cuda()
        sum1 = torch.zeros([self.num_claster]).cuda()
        sum2 = torch.zeros([self.num_claster, 10]).cuda()

        for x, target in self.train_loader:

            u = torch.zeros([self.network_list.__len__() + 1, self.batch_size, 10]).cuda()  #
            x = Variable(x).cuda()  #
            for i in range(0, self.network_list.__len__() + 1):

                self.AE_net[i].cuda()  #
                self.AE_net[i].eval()
                for param in self.AE_net[i].parameters():
                    param.requires_grad = False
            for kk in range(0, self.network_list.__len__() + 1):
                _, y = self.AE_net[kk](x)
                u[kk, :, :] = y.cuda()  #
                # print(torch.sum(torch.pow(y[0,:] - u_mean[kk, :], 2), dim=1))
            p = cluster_assignments(u, self.u_mean.cuda(), self.batch_size, self.num_claster, self.m, self.network_list)

            self.u_mean = self.u_mean.cuda()

            p = torch.pow(p, self.m)
            p = p.cuda()
            del u

            new_counter = 0
            for j in range(0, self.num_claster):
                if j in self.network_list:
                    _, u_cent[j, :, :] = self.AE_net[new_counter](x)
                    new_counter += 1
                else:
                    _, u_cent[j, :, :] = self.AE_net[-1](x)

            for kk in range(0, self.num_claster):
                sum1[kk] = sum1[kk] + torch.sum(p[:, kk])
                sum2[kk, :] = sum2[kk, :] + torch.matmul(p[:, kk].T, u_cent[kk])

            p = p.cuda()
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            target = target.numpy()

            true_label[ii * self.batch_size:(ii + 1) * self.batch_size] = target
            pred_label[ii * self.batch_size:(ii + 1) * self.batch_size] = y
            ii = ii + 1

        for kk in range(0, self.num_claster):
            self.u_mean[kk, :] = torch.div(sum2[kk, :], sum1[kk])

        print('nmi', nmi(true_label, pred_label))
        print('accuracy', acc(true_label, pred_label))

        return self.u_mean




    def train_DML(self):

        for i in range(0, self.network_list.__len__() + 1):
            self.AE_net[i].cuda()
            self.AE_net[i].train()
            for param in self.AE_net[i].parameters():
                param.requires_grad = True


        for x, target in self.train_loader:

            u = torch.zeros([self.network_list.__len__() + 1, self.batch_size, 10]).cuda()
            x = Variable(x).cuda()

            for kk in range(0, self.network_list.__len__() + 1):
                _, y = self.AE_net[kk](x)
                u[kk, :, :] = y.cuda()
            u = u.detach()
            p = cluster_assignments(u, self.u_mean.cuda(), self.batch_size, self.num_claster, self.m, self.network_list)

            p = p.detach()

            p = p.cuda()
            self.u_mean = self.u_mean.cuda()

            x = x.cuda()
            p = p.T

            p = torch.pow(p, self.m)


            for j in range(0, 1):
                network_counter = 0
                for i in range(0, self.num_claster):
                    if i in self.network_list:
                        y, u1 = self.AE_net[network_counter](x)

                        for jj in range(0, self.network_list.__len__() + 1):
                            if jj == network_counter:
                                continue
                        self.u_mean = self.u_mean.float()
                        [loss, a] = clustering_cost(x.view(-1, 784), y.view(-1, 784), u1,
                                                    p[i, :].unsqueeze(0),
                                                    self.u_mean[i, :].unsqueeze(0).repeat(self.batch_size, 1), self.landa)

                        self.optimizer[network_counter].zero_grad()
                        loss.backward()
                        self.optimizer[network_counter].step()
                        network_counter += 1
                    else:
                        y, u1 = self.AE_net[-1](x)
                        for jj in range(0, self.network_list.__len__() + 1):
                            if jj == self.network_list.__len__():
                                continue

                        self.u_mean = self.u_mean.float()
                        [loss, a] = clustering_cost(x.view(-1, 784), y.view(-1, 784), u1,
                                                    p[i, :].unsqueeze(0),
                                                    self.u_mean[i, :].unsqueeze(0).repeat(self.batch_size, 1), self.landa)

                        self.optimizer[-1].zero_grad()
                        loss.backward()
                        self.optimizer[-1].step()

        return self.AE_net, self.optimizer


    def train_main(self):
        for i in range(10):
            print("epoch:", i)
            self.u_mean = self.update_cluster_centers()
            self.AE_net, self.optimizer = self.train_DML()


