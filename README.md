# MMP
A PyTorch implementation of MMP "MEMORY-BASED MESSAGE PASSING: DECOUPLING THE MESSAGE FOR PROPAGATION FROM DISCRIMINATION
".(ICASSP 2022)

(https://arxiv.org/abs/2202.00423)

## Abstract
Message passing is a fundamental procedure for graph neural networks in the field of graph representation learning. Based on the homophily assumption, the current message passing always aggregates features of connected nodes, such as the graph Laplacian smoothing process. However, real-world graphs tend to be noisy and/or non-smooth. The homophily as- sumption does not always hold, leading to sub-optimal results. A revised message passing method needs to maintain each node’s discriminative ability when aggregating the message from neighbors. To this end, we propose a Memory-based Message Passing (MMP) method to decouple the message of each node into a self-embedding part for discrimination and a memory part for propagation. Furthermore, we develop a control mechanism and a decoupling regularization to con- trol the ratio of absorbing and excluding the message in the memory for each node. More importantly, our MMP is a gen- eral skill that can work as an additional layer to help improve traditional GNNs performance. Extensive experiments on var- ious datasets with different homophily ratios demonstrate the effectiveness and robustness of the proposed method.
## Dependencies
- python 3.7.3
- pytorch 1.6.0
- dgl 0.6.0
- torch-geometric 1.6.2

## Code Architecture
    |── datasets                # datasets and load scripts
    |── utils                   # Common useful modules(transform, loss function)
    |── models                  # models 
    |  └── layers               # code for layers
    |  └── models               # code for models
    └── train                   # train scripts
    

## Usage 
```
python train.py -d texas
python train.py -d wisconsin
python train.py -d actor
python train.py -d squirrel
python train.py -d chameleon
python train.py -d cornell
python train.py -d citeseer
python train.py -d pubmed
python train.py -d cora
```

## Citation
```
@article{chen2022memory,
  title={Memory-based Message Passing: Decoupling the Message for Propogation from Discrimination},
  author={Chen, Jie and Liu, Weiqi and Pu, Jian},
  journal={preprint arXiv:2202.00423},
  year={2022}
}
```