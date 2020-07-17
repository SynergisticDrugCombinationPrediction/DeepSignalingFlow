import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, add_self):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self = add_self
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        
    def forward(self, x, adj, inv_deg):
        # pdb.set_trace()
        adj = torch.transpose(adj, 1, 2)
        new_x = torch.matmul(adj, x)
        new_x = torch.matmul(inv_deg, new_x)
        if self.add_self == 'add':
            new_x += x
        new_x = torch.matmul(new_x, self.weight)
        new_x = new_x + torch.matmul(x, self.bias)
        new_x = F.normalize(new_x, p = 2, dim = 2)
        return new_x

class GcnDecoderGraph(nn.Module):
    def __init__(self, add_self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_layer):
        super(GcnDecoderGraph, self).__init__()
        self.num_layer = num_layer
        self.node_num = node_num
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer)
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)
        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim, decoder_dim))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim))

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain = nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.xavier_uniform(m.bias.data, gain = nn.init.calculate_gain('relu'))

    def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer):
        # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
        conv_first = GraphConv(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self)
        # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
        conv_block = nn.ModuleList([GraphConv(input_dim = hidden_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self) 
                    for i in range(num_layer - 2)])
        # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
        conv_last = GraphConv(input_dim = hidden_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self)
        return conv_first, conv_block, conv_last

    def forward(self, x, adj, inv_deg):
        # conv
        # pdb.set_trace()
        x = self.conv_first(x, adj, inv_deg)
        x = self.act2(x)
        for i in range(self.num_layer - 2):
            x = self.conv_block[i](x, adj, inv_deg)
            x = self.act2(x)
        x = self.conv_last(x, adj, inv_deg)
        x = self.act2(x)
        drug_a_embedding = x[:, self.node_num - 2]
        drug_b_embedding = x[:, self.node_num - 1]
        ypred = torch.zeros(x.shape[0], 1)
        for i in range(x.shape[0]):
            product1 = torch.matmul(drug_a_embedding[i], self.parameter1)
            product2 = torch.matmul(product1, self.parameter2)
            product3 = torch.matmul(product2, torch.transpose(self.parameter1, 0, 1))
            output = torch.matmul(product3, drug_b_embedding[i].reshape(-1, 1))
            ypred[i] = output
        # print(self.parameter1)
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        loss = F.mse_loss(pred.squeeze(), label)
        return loss