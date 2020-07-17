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
        self.weight = torch.FloatTensor(input_dim, output_dim)
        self.bias = torch.FloatTensor(input_dim, output_dim)
        self.weight_adj = torch.FloatTensor(node_num, node_num)
        
    def forward(self, x, adj, inv_deg):
        # ADDING ADJ_WEIGHT MATRIX

        # adj = torch.mul(adj, self.weight_adj)
        # pdb.set_trace()
        new_x = torch.matmul(adj, x)
        new_x = torch.matmul(inv_deg, new_x)
        if self.add_self == 'add':
            new_x += x
        new_x = torch.matmul(new_x, self.weight)
        new_x = new_x + torch.matmul(x, self.bias)
        new_x = F.normalize(new_x, p = 2, dim = 2)
        return new_x

class GcnEncoderGraph(nn.Module):
    def __init__(self, add_self, input_dim, hidden_dim, embedding_dim, node_num,
                    label_dim, num_layer, pred_hidden_dim_list):
        super(GcnEncoderGraph, self).__init__()
        self.num_layer = num_layer

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        self.concat = True
        concat = self.concat
        # CONCAT ALL CONVOLUTION LAYERS ON ALL [conv_first, conv_block, conv_last]
        if concat:
            self.pred_input_dim = hidden_dim * (num_layer - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layer(self.pred_input_dim, pred_hidden_dim_list, label_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain = nn.init.calculate_gain('relu'))
                m.weight_adj.data = init.xavier_uniform(m.weight_adj.data, gain = nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer):
        # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
        conv_first = GraphConv(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self)
        # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
        conv_block = nn.ModuleList([GraphConv(input_dim = hidden_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self) 
                    for i in range(num_layer - 2)])
        # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
        conv_last = GraphConv(input_dim = hidden_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self)
        return conv_first, conv_block, conv_last

    def build_pred_layer(self, pred_input_dim, pred_hidden_dim_list, label_dim):
        # [pred_hidden_dim_list] SHOW POOLING
        # IF [pred_hidden_dim_list=[]] USE NO POOLING 
        # ELSE USE POOLING e.g.[pred_hidden_dim_list=[120, 20, 10]]
        if len(pred_hidden_dim_list) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            # USE FOR LOOP TO DO POOLING [nn.Linear(pred_input_dim, pred_dim)] ADDING ON [pred_layers]
            pred_layers = []
            for pred_dim in pred_hidden_dim_list:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model


    def forward(self, x, adj, inv_deg):
        # conv
        # pdb.set_trace()
        x = self.conv_first(x, adj, inv_deg)
        # x = self.act(x)
        # if self.bn:
        #     x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim = 1)
        out_all.append(out)
        for i in range(self.num_layer - 2):
            x = self.conv_block[i](x, adj, inv_deg)
            # x = self.act(x)
            # if self.bn:
            #     x = self.apply_bn(x)
            out,_ = torch.max(x, dim = 1)
            out_all.append(out)
        x = self.conv_last(x, adj, inv_deg)
        #x = self.act(x)
        out, _ = torch.max(x, dim = 1)
        out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim = 1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label):
        print('COMPUTING LOSS...')
        loss = F.mse_loss(pred.squeeze(), label)
        return loss

