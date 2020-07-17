import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, add_self, dropout):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p = dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.add_self = add_self
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim).cuda())
        self.bias = torch.nn.Parameter(torch.randn(input_dim, output_dim).cuda())

        self.weight_adj = torch.nn.Parameter(torch.randn(node_num, node_num).cuda())
        init.xavier_uniform(self.weight_adj.data, gain = nn.init.calculate_gain('relu'))
        
    def forward(self, x, adj, inv_deg):
        # OPTION ON USING DRUG NODES IN ADJACENT MATRICES
        gene_adj = True
        adj_act = False
        if gene_adj == True:
            # GENE NODES ADJACENT MATRIX WITH MASK
            mask_adj = torch.ones(self.node_num, self.node_num).cuda()
            mask_adj[:, -2:] = 0.0
            mask_adj[-2:, :] = 0.0
            weighted_adj = torch.mul(adj, self.weight_adj)
            mask_weighted_adj = torch.mul(weighted_adj, mask_adj)
            # ORIGINAL DRUG GENE ADAJCENT MATRIX WITH MASK
            unmask_adj = torch.zeros(self.node_num, self.node_num).cuda()
            unmask_adj[:, -2:] = 1.0
            unmask_adj[-2:, :] = 1.0
            drug_adj = torch.mul(adj, unmask_adj)
            # COMBINE TRAINED WEIGHTED GENE EDGES AND DRUG-GENE EDGES
            new_adj = mask_weighted_adj + drug_adj
        else:
            new_adj = torch.mul(adj, self.weight_adj)
        # OPTION ON [LeakyReLU] TO ADD NON-LINEARITY FOR EDGE WEIGHTS
        if adj_act == True:
            adj = self.act2(new_adj)
        else:
            adj = new_adj

        # DECISION ON USING DROPOUT
        gene_node = True
        if self.dropout > 0.001:
            # OPTION ON DROPING OUT DRUG NODES
            if gene_node == True:
                mask_x = torch.zeros(self.node_num, self.input_dim).cuda()
                mask_x[:-2, :] = 1.0
                gene_x = torch.mul(x, mask_x)
                gene_x = self.dropout_layer(gene_x)
                unmask_x = torch.ones(self.node_num, self.input_dim).cuda()
                unmask_x[:-2, :] = 0.0
                drug_x = torch.mul(x, unmask_x)
                x = gene_x + drug_x
            else:
                x = self.dropout_layer(x)
        
        adj = torch.transpose(adj, 1, 2)
        new_x = torch.matmul(adj, x)
        new_x = torch.matmul(inv_deg, new_x)
        new_x = torch.matmul(new_x, self.weight)
        self_x = torch.matmul(x, self.bias)
        concat_x = torch.cat((new_x, self_x), 2)
        concat_x = F.normalize(concat_x, p = 2, dim = 2)
        return concat_x


class GraphSAGEDecoder(nn.Module):
    def __init__(self, add_self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_layer, dropout):
        super(GraphSAGEDecoder, self).__init__()
        self.num_layer = num_layer
        self.node_num = node_num
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout)
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)
        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim * 2, decoder_dim).cuda())
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).cuda())


        for m in self.modules():
            if isinstance(m, GraphSAGE):
                m.weight.data = init.xavier_uniform(m.weight.data, gain = nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.xavier_uniform(m.bias.data, gain = nn.init.calculate_gain('relu'))


    # # OLD CONVOLUTION LAYERS
    # def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout = 0.0):
    #     # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
    #     conv_first = GraphSAGE(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #     next_input_dim = input_dim + hidden_dim
    #     # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
    #     conv_block = nn.ModuleList()
    #     for i in range(num_layer - 2):
    #         conv_block_layer = GraphSAGE(input_dim = next_input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #         conv_block.append(conv_block_layer)
    #         next_input_dim = hidden_dim * 2
    #     # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
    #     conv_last = GraphSAGE(input_dim = next_input_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #     return conv_first, conv_block, conv_last

    # NOVEL ADDITION CONVOLUTION LAYERS
    def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout):
        # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
        conv_first = GraphSAGE(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
        next_input_dim = input_dim + hidden_dim
        # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
        conv_block = nn.ModuleList()
        for i in range(num_layer - 2):
            conv_block_layer = GraphSAGE(input_dim = next_input_dim, output_dim = next_input_dim, node_num = node_num, add_self = add_self, dropout = dropout)
            conv_block.append(conv_block_layer)
            next_input_dim = next_input_dim * 2
        # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
        conv_last = GraphSAGE(input_dim = next_input_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self, dropout = dropout)
        return conv_first, conv_block, conv_last


    def forward(self, x, adj, inv_deg):
        # pdb.set_trace()
        # [GraphSAGE] CONVOLUTION LAYERS
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
        pred = pred.cuda()
        label = label.cuda()
        loss = F.mse_loss(pred.squeeze(), label)
        return loss