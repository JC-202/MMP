import torch
import sys

from torch_sparse import SparseTensor

sys.path.append('..')

import warnings
from .layers import FC_Layer, GCNConv, GATConv

warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F

class Gate_Memory(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Gate_Memory, self).__init__()
        self.input_dim = input_dim
        self.gate_dim = 1
        self.linear = FC_Layer(input_dim * 2, 4 * self.gate_dim, activation=None, dropout=dropout, bias=True)

    def forward(self, h_s, h_m):
        combined = torch.cat([h_s, h_m], dim=1)
        combined_conv = self.linear(combined)

        #ignore the h_s when update h_m
        a_h, a_m, _, a_c = torch.split(combined_conv.sigmoid(), self.gate_dim, dim=1)
        h_s = h_s * a_h + h_m * a_m
        h_m = h_m * a_c
        return h_s, h_m


class MMP(nn.Module):
    def __init__(self, g, in_feats, hid_feats, out_feats, num_layers=2, activation=F.relu,
                 conv_type='gcn', dropout=0, num_head=1, reg_type=None):
        super(MMP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.num_head = num_head
        self.dropout = dropout
        self.conv_type = conv_type

        self.input_fc = FC_Layer(in_feats, hid_feats, activation, dropout, bias=True)
        self.build_message_passing_layers(num_layers, hid_feats)
        self.classifier = FC_Layer(hid_feats, out_feats, None, 0, bias=True)
        self.gate = Gate_Memory(hid_feats, self.dropout)

        self.reg_type = reg_type
        self.g = self.set_g(g)


    def set_g(self, adj):
        if self.conv_type == 'gat':
            if isinstance(adj, SparseTensor):
                row, col, _ = adj.coo()
                adj = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
            elif isinstance(adj, torch.Tensor):
                if adj.shape[0] == adj.shape[1]:
                    adj = adj.nonzero().t()
        return adj

    def build_message_passing_layers(self, num_layers, hid_channesl):
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if self.conv_type == 'gcn':
                layer = GCNConv(hid_channesl, hid_channesl)
            elif self.conv_type == 'gat':
                layer = GATConv(hid_channesl, hid_channesl, self.dropout, 0, 0.2, self.num_head, True)
            else:
                layer = FC_Layer(hid_channesl, hid_channesl, self.activation, self.dropout, bias=True)
            self.convs.append(layer)

    def propogate(self, conv, graph, h):
        m = conv(graph, h)
        return m

    def forward(self, graph, inputs):
        graph = self.g
        # used for computing decoupling regularization
        h_list, memory_list = [], []

        h = self.input_fc(inputs)
        memory = h
        for i in range(self.num_layers):
            cell = self.propogate(self.convs[i], graph, memory)
            h, memory = self.gate(h, cell)
            h_list.append(h)
            memory_list.append(memory)
        out = self.classifier(h)

        # compute decoupling regularization loss
        self.reg_loss = self.memory_loss(h_list, memory_list)
        return out

    def memory_loss(self, h_list, memory_list):
        reg_loss = torch.tensor(0.0).to(h_list[0].device)
        if self.reg_type == 'cos':
            for i, (h, memory) in enumerate(zip(h_list, memory_list)):
                decouple_reg = torch.abs(torch.cosine_similarity(h, memory).mean())
                reg_loss += decouple_reg
            if self.num_layers > 1:
                reg_loss = reg_loss / (self.num_layers - 1)
        return reg_loss

    def memory_loss1(self, h, memory):
        reg = 0
        if self.reg_type == 'cos':
            reg = torch.abs(torch.cosine_similarity(h, memory).mean())
        return reg

