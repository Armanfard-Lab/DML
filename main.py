from torchvision import datasets, transforms
import heapq
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import copy

from DML import DML
from Autoencoder import AutoEncoder,myBottleneck
from utils import acc

batch_size = 1000
dataset_size = 70000
num_clusters = 10
train_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=True,
                       transform=transforms.ToTensor(), download=True)
test_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=False,
                      transform=transforms.ToTensor(), download=True)

kwargs = {'num_workers': 1}

train1 = torch.utils.data.ConcatDataset([train_set, test_set])
# train1 = test_set
train_loader = torch.utils.data.DataLoader(
    dataset=train1,
    batch_size=batch_size,
    shuffle=True, **kwargs)




if __name__ == '__main__':
    ################################################# Hyperparameters


    m = 1.5
    landa = 0.1
    ################################################# loading AE
    AE = torch.load('AE_DSL_MNIST')
    u_mean = torch.load("Centers_DSL_MNIST")

    ################################################# initializing number of AEs
    temp_max = 0.0
    temp_min = 1e7
    for mm in range(0, num_clusters):
      for tt in range(mm + 1, num_clusters):
          a = torch.sum(torch.pow(u_mean[mm, :] - u_mean[tt, :], 2))
          a = torch.sqrt(a)
          if temp_max < a:
             temp_max = a
          if temp_min > a:
             temp_min = a

    new_list = []
    for mm in range(0, num_clusters):
        for tt in range(mm + 1, num_clusters):
            a = torch.sum(torch.pow(u_mean[mm, :] - u_mean[tt, :], 2))
            a = torch.sqrt(a)
            if a < (temp_min + temp_max) / 4:
                new_list.append(tt)
                new_list.append(mm)

    network_list = []
    for x in new_list:
        if x not in network_list:
            network_list.append(x)
    print(network_list)
    del new_list

    AE_net = []
    optimizer = []
    for i in range(0, network_list.__len__() + 1):
        AE_net.append(copy.deepcopy(AE))

    for i in range(0, network_list.__len__() + 1):
        optimizer.append(optim.Adam(AE_net[i].parameters(), lr=0.00001))
#########################################################################################

    dml = DML(AE_net, train_loader, batch_size, optimizer, u_mean, num_clusters, m, network_list, dataset_size, landa)
    dml.train_main()
