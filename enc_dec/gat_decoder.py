import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, add_self):
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.add_self = add_self
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        
    def forward(self, x, adj, inv_deg):
        adj = torch.transpose(adj, 1, 2)
        new_x = torch.matmul(adj, x)
        new_x = torch.matmul(inv_deg, new_x)
        new_x = torch.matmul(new_x, self.weight)
        self_x = torch.matmul(x, self.bias)
        concat_x = torch.cat((new_x, self_x), 2)
        concat_x = F.normalize(concat_x, p = 2, dim = 2)
        return concat_x


class GATDecoder(nn.Module):
    def __init__(self, add_self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_layer):
        super(GATDecoder, self).__init__()
        self.num_layer = num_layer
        self.node_num = node_num
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer)
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)

        self.input_dim = input_dim
        self.W = nn.Parameter(torch.zeros(size = (input_dim, input_dim)))
        nn.init.xavier_uniform_(self.W.data, gain = nn.init.calculate_gain('relu'))
        self.a = nn.Parameter(torch.zeros(size = (2 * input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = nn.init.calculate_gain('relu'))
        self.act3 = nn.LeakyReLU(negative_slope = 0.2)

        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim * 2, decoder_dim))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim))


        for m in self.modules():
            if isinstance(m, GraphSAGE):
                m.weight.data = init.xavier_uniform(m.weight.data, gain = nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.xavier_uniform(m.bias.data, gain = nn.init.calculate_gain('relu'))


    def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer):
        # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
        conv_first = GraphSAGE(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self)
        next_input_dim = input_dim + hidden_dim
        # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
        conv_block = nn.ModuleList()
        for i in range(num_layer - 2):
            conv_block_layer = GraphSAGE(input_dim = next_input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self)
            conv_block.append(conv_block_layer)
            next_input_dim = hidden_dim * 2
        # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
        conv_last = GraphSAGE(input_dim = next_input_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self)
        return conv_first, conv_block, conv_last


    def forward(self, x, adj, inv_deg):
        # FORM ATTENTION ADJACENT MATRIX
        batch_size = x.size()[0]
        h = torch.matmul(x, self.W)
        N = h.size()[1]

        row_h = h.repeat(1, 1, N).reshape(batch_size, N * N, self.input_dim)
        col_h = h.repeat(1, N, 1).reshape(batch_size, N * N, self.input_dim)
        a_input = torch.cat([row_h, col_h], dim = 2).reshape(batch_size, N, N, 2 * self.input_dim)
        e = self.act3(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim = 2)
        adj = attention

        # pdb.set_trace()
        # CONTINUE GRAPH CONVOLUTION
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
        print(self.parameter1)
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        loss = F.mse_loss(pred.squeeze(), label)
        return loss