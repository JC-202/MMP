import torch
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..')
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl

class FC_Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, activation, dropout, use_res=False, use_bn=False, bias=True ):
        super(FC_Layer, self).__init__()
        self.linear = nn.Linear(in_dim, hid_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = use_res
        self.bias = bias
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(hid_dim)
        self.use_bn = use_bn

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            nn.init.constant(self.linear.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        res = x

        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual:
            x += res

        return x


class GCNConv(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hid_dim)

    def forward(self, adj, h):
        h = adj @ h
        h = self.fc(h)
        return h


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    # def special_spmm(self, indices, values, shape, b):
    #     a = torch.sparse_coo_tensor(indices, values, shape)
    #     return a @ b

    def edge_softmax(self, adj, e):
        g = dgl.graph((adj[0, :], adj[1, :]))
        return dgl.nn.functional.edge_softmax(g, e)

    def forward(self, adj, input):
        dv = input.device
        N = input.size()[0]
        h = self.fc(input)
        # h: N x out
        if torch.isnan(h).any():
            print(h)
            print(self.W)
            print(self.W.grad)
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[adj[0, :], :], h[adj[1, :], :]), dim=1).t()
        # edge: 2*D x E

        e = self.leakyrelu(self.a.mm(edge_h).squeeze())
        assert not torch.isnan(e).any()
        # edge_e: E

        attention = self.edge_softmax(adj, e)
        attention = self.dropout(attention)

        sparse_attention = SparseTensor(row=adj[0, :], col=adj[1, :], value=attention, sparse_sizes=(N, N)).to(dv)
        h_prime = sparse_attention @ h
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GATConv(nn.Module):
    def __init__(self, nfeat, nhid, dropout, drop_edge, alpha, nheads, concat=True):
        """Sparse version of GAT."""
        super().__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=drop_edge,
                                                 alpha=alpha,
                                                 concat=concat) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)
        x = F.elu(x)
        return x