import torch
import numpy as np
import networkx as nx
import torch.utils.data

class BiGraphLoader():
    def __init__(self):
        pass

    def load_graph(graph_attribute_list, graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list, graph_label_list, args):
        dataset_sampler = BiGraphSampler(graph_attribute_list, graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list, graph_label_list)
        dataset_loader = torch.utils.data.DataLoader(
                    dataset = dataset_sampler, 
                    batch_size = args.batch_size, 
                    shuffle = False,
                    num_workers = args.num_workers)
        return dataset_loader, dataset_sampler.node_num, dataset_sampler.feature_dim

# CUSTOM GRAPH DATASET WITH [init, len, getitem]
class BiGraphSampler(torch.utils.data.Dataset):
    # INITIATE WITH [node_num, feature_dim, adj_all, inv_deg_all, label_all, feature_all] FROM [graph_list]
    def __init__(self, graph_attribute_list, graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list, graph_label_list):
        self.node_num = graph_adj_list[0].shape[0]
        self.feature_dim = graph_attribute_list[0].shape[1]
        self.adj_all = graph_adj_list
        self.up_inv_deg_all = graph_up_inv_deg_list
        self.down_inv_deg_all = graph_down_inv_deg_list
        self.feature_all = graph_attribute_list
        self.label_all = graph_label_list

    def __len__(self):
        return len(self.label_all)

    # INPUT POINT: A GRAPH (ALL OF NODES, AND ITS NODES FEATURES)
    # INPUT LABEL: A GRAPH LABEL
    def __getitem__(self, idx):
        return {'adj': self.adj_all[idx],
                'up_inv_deg': self.up_inv_deg_all[idx],
                'down_inv_deg': self.down_inv_deg_all[idx],
                'feature': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                'num_node': self.node_num}