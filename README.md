# DML: Deep Multi-Representation Learning

PyTorch implementation of DML.

<center><img src="https://github.com/Armanfard-Lab/DML/blob/main/Figs/model-1.png" alt="Overview" width="800" align="center"></center>

## Citation

Please cite our paper if you use the results of our work.

```
@article{sadeghi2022deep,
  title={Deep Multi-Representation Learning for Data Clustering},
  author={Sadeghi, Mohammadreza and Armanfard, Narges},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
```

## Abstract

>Deep clustering incorporates embedding into clustering in order to find a lower-dimensional space suitable for clustering task. Conventional deep clustering methods aim to obtain a single global embedding subspace (aka latent space) for all the data clusters. In contrast, in this paper, we propose a deep multi-representation learning (DML) framework for data clustering whereby each difficult to cluster data group is associated with its own distinct optimized latent space, and all the easy to cluster data groups are associated to a general common latent space. Autoencoders are employed for generating the cluster-specific and general latent spaces. To specialize each autoencoder in its associated data cluster(s), we propose a novel and effective loss function which consists of weighted reconstruction and clustering losses of the data points, where higher weights are assigned to the samples more probable to belong to the corresponding cluster(s). Experimental results on benchmark datasets demonstrate that the proposed DML framework and loss function outperform state-of-the-art clustering approaches. In addition, the results show that the DML method significantly outperforms the SOTA on imbalanced datasets as a result of assigning an individual latent space to the difficult clusters.

