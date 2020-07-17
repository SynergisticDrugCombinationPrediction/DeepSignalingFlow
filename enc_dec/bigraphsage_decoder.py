import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class BiGraphSAGE(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, add_self, dropout):
        super(BiGraphSAGE, self).__init__()
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p = dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.add_self = add_self
        self.up_weight = torch.nn.Parameter(torch.randn(input_dim, output_dim).cuda())
        self.down_weight = torch.nn.Parameter(torch.randn(input_dim, output_dim).cuda())
        self.bias = torch.nn.Parameter(torch.randn(input_dim, output_dim).cuda())

        self.up_weight_adj = torch.nn.Parameter(torch.randn(node_num, node_num).cuda())
        init.xavier_uniform(self.up_weight_adj.data, gain = nn.init.calculate_gain('relu'))

        self.down_weight_adj = torch.nn.Parameter(torch.randn(node_num, node_num).cuda())
        init.xavier_uniform(self.down_weight_adj.data, gain = nn.init.calculate_gain('relu'))
        
    def forward(self, x, adj, up_inv_deg, down_inv_deg):
        # GENE NODES ADJACENT MATRIX WITH MASK
        mask_adj = torch.ones(self.node_num, self.node_num).cuda()
        mask_adj[:, -2:] = 0.0
        mask_adj[-2:, :] = 0.0
        up_weighted_adj = torch.mul(adj, self.up_weight_adj)
        up_gene_adj = torch.mul(up_weighted_adj, mask_adj)
        down_weighted_adj = torch.mul(adj, self.down_weight_adj)
        down_gene_adj = torch.mul(down_weighted_adj, mask_adj)
        # ORIGINAL DRUG GENE ADAJCENT MATRIX WITH MASK
        unmask_adj = torch.zeros(self.node_num, self.node_num).cuda()
        unmask_adj[:, -2:] = 1.0
        unmask_adj[-2:, :] = 1.0
        up_drug_adj = torch.mul(adj, unmask_adj)
        down_drug_adj = torch.mul(adj, unmask_adj)
        # COMBINE TRAINED WEIGHTED GENE EDGES AND DRUG-GENE EDGES
        up_adj = up_gene_adj + up_drug_adj
        down_adj = down_gene_adj + down_drug_adj

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

        # FROM UPSTREAM TO DOWNSTREAM WITH ORIGINAL IN-DEGREE ADJACENT
        up_adj = torch.transpose(up_adj, 1, 2)
        up_x = torch.matmul(up_adj, x)
        up_x = torch.matmul(up_inv_deg, up_x)
        # FROM DOWNSTREAM TO UPSTREAM WITH ORGINAL OUT-DEGREE ADJACENT
        down_x = torch.matmul(down_adj, x)
        down_x = torch.matmul(down_inv_deg, down_x)

        # pdb.set_trace()

        up_x = torch.matmul(up_x, self.up_weight)
        down_x = torch.matmul(down_x, self.down_weight)
        self_x = torch.matmul(x, self.bias)
        concat_x = torch.cat((up_x, down_x, self_x), 2)
        concat_x = F.normalize(concat_x, p = 2, dim = 2)
        return concat_x


class BiGraphSAGEDecoder(nn.Module):
    def __init__(self, add_self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_layer, dropout):
        super(BiGraphSAGEDecoder, self).__init__()
        self.num_layer = num_layer
        self.node_num = node_num
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout)
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)
        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim * 3, decoder_dim).cuda())
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).cuda())


        for m in self.modules():
            if isinstance(m, BiGraphSAGE):
                m.up_weight.data = init.xavier_uniform(m.up_weight.data, gain = nn.init.calculate_gain('relu'))
                m.down_weight.data = init.xavier_uniform(m.down_weight.data, gain = nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.xavier_uniform(m.bias.data, gain = nn.init.calculate_gain('relu'))

    # OLD CONVOLUTION LAYERS
    def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout = 0.0):
        # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
        conv_first = BiGraphSAGE(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
        next_input_dim = input_dim + hidden_dim * 2
        # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
        conv_block = nn.ModuleList()
        for i in range(num_layer - 2):
            conv_block_layer = BiGraphSAGE(input_dim = next_input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
            conv_block.append(conv_block_layer)
            next_input_dim = hidden_dim * 3
        # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
        conv_last = BiGraphSAGE(input_dim = next_input_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self, dropout = dropout)
        return conv_first, conv_block, conv_last

    # # NOVEL ADDITION CONVOLUTION LAYERS
    # def build_conv_layer(self, add_self, input_dim, hidden_dim, embedding_dim, node_num, num_layer, dropout):
    #     # FIRST CONVOLUTION LAYER [input_dim, hidden_dim, node_num, add_self]
    #     conv_first = BiGraphSAGE(input_dim = input_dim, output_dim = hidden_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #     next_input_dim = input_dim + hidden_dim * 2
    #     # BUILD [num_layer - 2] CONVOLUTION LAYER [hidden_dim, hidden_dim, dropout, node_num, add_self]
    #     conv_block = nn.ModuleList()
    #     for i in range(num_layer - 2):
    #         conv_block_layer = BiGraphSAGE(input_dim = next_input_dim, output_dim = next_input_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #         conv_block.append(conv_block_layer)
    #         next_input_dim = next_input_dim * 3
    #     # LAST CONVOLUTION LAYER [hidden_dim, embedding_dim, node_num, add_self]
    #     conv_last = BiGraphSAGE(input_dim = next_input_dim, output_dim = embedding_dim, node_num = node_num, add_self = add_self, dropout = dropout)
    #     return conv_first, conv_block, conv_last


    def forward(self, x, adj, up_inv_deg, down_inv_deg):
        # pdb.set_trace()
        # [BiGraphSAGE] CONVOLUTION LAYERS
        x = self.conv_first(x, adj, up_inv_deg, down_inv_deg)
        x = self.act2(x)
        for i in range(self.num_layer - 2):
            x = self.conv_block[i](x, adj, up_inv_deg, down_inv_deg)
            x = self.act2(x)
        x = self.conv_last(x, adj, up_inv_deg, down_inv_deg)
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