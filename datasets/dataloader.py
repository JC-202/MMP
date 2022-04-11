import torch
import sys
sys.path.append('.')
from dgl.data import *
from torch_geometric.data import Data
import dgl
import numpy as np
from utils.graph_transform import normalize_adj, normalizeLelf, sparse_normalize_adj_left, remove_self_loop
from torch_sparse import SparseTensor
from os import path as path
import scipy
import scipy.io

class MMPData():
    def __init__(self, pyg_data, device, to_dense=False, to_dgl=False):
        self.edge_index = pyg_data.edge_index
        self.x = pyg_data.x
        self.y = pyg_data.y
        self.num_of_class = pyg_data.y.max().item()+1
        self.num_of_nodes = self.x.shape[0]
        self.name = pyg_data.name
        self.id_mask = torch.ones(self.x.shape[0]).bool().to(device)
        self.train_mask = pyg_data.train_mask
        self.val_mask = pyg_data.val_mask
        self.test_mask = pyg_data.test_mask
        self.device = device
        self.init_adj(pyg_data.edge_index, to_dense, to_dgl)
        self.init_mask()

    def init_adj(self, edge_index, to_dense, to_dgl):
        self.noself_edge_index = remove_self_loop(edge_index)
        self.self_edge_index = torch.stack([torch.arange(self.x.shape[0]), torch.arange(self.x.shape[0])]).to(
            self.device)
        num_of_nodes = self.num_of_nodes
        adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(num_of_nodes, num_of_nodes))
        self.adj = adj.to(self.device)
        self.degree = adj @ torch.ones(self.x.shape[0]).to(self.device).view(-1, 1)

        self.eye = SparseTensor(row=torch.arange(self.x.shape[0]), col=torch.arange(self.x.shape[0]), sparse_sizes=(num_of_nodes, num_of_nodes)).to(self.device)
        self.norm_adj = sparse_normalize_adj_left(adj)
        self.noself_adj = SparseTensor(row=self.noself_edge_index[0, :], col=self.noself_edge_index[1, :], sparse_sizes=(num_of_nodes, num_of_nodes))
        self.norm_noself_adj = sparse_normalize_adj_left(self.noself_adj)
        if to_dense:
            self.adj = self.adj.to_dense().to(self.device)
            self.norm_adj = self.norm_adj.to_dense().to(self.device)
            self.noself_adj = self.noself_adj.to_dense().to(self.device)
            self.norm_noself_adj = self.norm_noself_adj.to_dense().to(self.device)
        if to_dgl:
            self.edge_index = dgl.graph((self.edge_index[0], self.edge_index[1])).to(self.device)

    def init_mask(self):
        if self.name in ['cora', 'citeseer', 'pubmed']:
            self.train_mask = torch.arange(self.x.shape[0])[self.train_mask].to(self.device)
            self.val_mask = torch.arange(self.x.shape[0])[self.val_mask].to(self.device)
            self.test_mask = torch.arange(self.x.shape[0])[self.test_mask].to(self.device)

    def get_dgl(self):
        g = dgl.graph((self.edge_index[0], self.edge_index[1])).to(self.device)
        return g

DATAPATH = path.dirname(path.abspath(__file__))


def load_data(data_name, device, split_id=0, to_dense=False, to_dgl=False,):
    data = load_pyg_data(data_name, device, split_id=split_id)
    data = MMPData(data, device, to_dense, to_dgl, )
    return data


def load_dgl_data(data_name, device='cpu', split_id=0):
    if data_name in ['cora', 'citeseer', 'pubmed']:
        if data_name == 'cora':
            dataset = CoraGraphDataset(verbose=False)
        elif data_name == 'citeseer':
            dataset = CiteseerGraphDataset(verbose=False)
        elif data_name == 'pubmed':
            dataset = PubmedGraphDataset(verbose=False)
        g = dataset[0]
        features, labels, train_mask, val_mask, test_mask = g.ndata['feat'], g.ndata['label'], g.ndata[
            'train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        splits, train_mask, val_mask, test_mask = load_pyg_splits(data_name, split_id)
    else:
        data = load_pyg_he_data(data_name, split_id)
        dgl_graph = pyg2dgl(data, device)
        g, features, labels, train_mask, val_mask, test_mask = dgl_graph, dgl_graph.x, dgl_graph.y, dgl_graph.train_mask, dgl_graph.val_mask, dgl_graph.test_mask
    g = g.remove_self_loop()
    g = g.add_self_loop()
    return g.int().to(device), features.to(device), labels.to(device), train_mask.to(device), val_mask.to(
        device), test_mask.to(device)

def load_pyg_data(data_name, device, split_id=0):
    g, features, labels, train_mask, val_mask, test_mask = load_dgl_data(data_name, device, split_id)
    edges = [a.long() for a in g.edges()]
    data = Data(x=features, y=labels, edge_index=torch.stack(edges))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.name = data_name
    return data


def load_pyg_he_data(dataset_name, split_id=0):
    if dataset_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        hedata = scipy.io.loadmat(f'{DATAPATH}/hedata/{dataset_name}.mat')
        edge_index = hedata['edge_index']
        node_feat = hedata['node_feat']
        label = np.array(hedata['label'], dtype=np.int).flatten()
        data = Data(x=torch.tensor(node_feat),
                    y=torch.tensor(label, dtype=torch.long),
                    edge_index=edge_index)
        data.split_idxs, data.train_mask, data.val_mask, data.test_mask = load_pyg_splits(dataset_name, split_id)
        return data
    else:
        raise 'not implement pyg dataset'

def pyg2dgl(pyg_data, device='cpu'):
    data = dgl.DGLGraph()
    data.add_edges(pyg_data.edge_index[0], pyg_data.edge_index[1])
    data.x, data.y = pyg_data.x, pyg_data.y
    data.train_mask, data.val_mask, data.test_mask = pyg_data.train_mask, pyg_data.val_mask, pyg_data.test_mask
    return data.to(device)


def load_pyg_splits(dataset_name, split_id=0):
    name = dataset_name
    split_path = path.dirname(path.abspath(__file__))
    file_path = f'{split_path}/hedata/splits/{name}-splits.npy'
    splits_lst = np.load(file_path, allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    train_mask = splits_lst[split_id]['train']
    val_mask = splits_lst[split_id]['valid']
    test_mask = splits_lst[split_id]['test']
    return splits_lst, train_mask, val_mask, test_mask
