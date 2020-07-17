import numpy as np
import pandas as pd
import networkx as nx

from numpy import inf

class ReadBiGraph():
    def __init__(self, dir_opt, final_data_path, RNA_seq_filename, form_data_path):
        self.dir_opt = dir_opt
        self.final_data_path = final_data_path
        self.RNA_seq_filename = RNA_seq_filename
        self.form_data_path = form_data_path

    def read_attribute(self, num_gene, num_feature):
        dir_opt = self.dir_opt
        final_data_path = self.final_data_path
        RNA_seq_filename = self.RNA_seq_filename
        form_data_path = self.form_data_path
        xBatch = np.load(form_data_path + '/xBatch.npy')
        # ADD [0, 0, 0] FOR DRUG NODES
        num_graph = xBatch.shape[0]
        drug_feature = np.zeros((num_graph, 2 * num_feature))
        xBatch = np.hstack((xBatch, drug_feature))
        # FORM [graph_attribute_list]
        num_node = num_gene + 2
        xBatch = xBatch.reshape(num_graph, num_node, num_feature)
        graph_attribute_list = []
        for i in range(num_graph):
            graph_attribute_list.append(xBatch[i, :, :])
        return graph_attribute_list

    def read_adjacent(self, num_gene, num_feature, args):
        dir_opt = self.dir_opt
        final_data_path = self.final_data_path
        RNA_seq_filename = self.RNA_seq_filename
        form_data_path = self.form_data_path
        # GENE-GENE ADJACENT MATRIX
        cellline_gene_num_df = pd.read_csv(form_data_path + '/gene_connection_num.txt')
        src_gene_list = list(cellline_gene_num_df['src'])
        dest_gene_list = list(cellline_gene_num_df['dest'])
        num_node = num_gene + 2
        gene_adj = np.zeros((num_node, num_node))
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            gene_adj[row_idx, col_idx] = 1
            if args.adj == 'sym':
                gene_adj[col_idx, row_idx] = 1
        # GENE-GENE DEGREE MATRIX
        up_inv_deg_matrix, down_inv_deg_matrix = bigraph_degree(gene_adj)
        # for i in range(gene_adj.shape[0]):
        #     if np.sum(gene_adj[i]) == 0:
        #         print('YES', i)
        #         print(np.sum(gene_adj[i]))
        # FORM UNIQUE [graph_adj_list] ACCORDING DIFFERENT DRUG-GENE ATTR
        xBatch = np.load(form_data_path + '/xBatch.npy')
        num_graph = xBatch.shape[0]
        xBatch = xBatch.reshape(num_graph, num_gene, num_feature)
        graph_adj_list = []
        graph_up_inv_deg_list = []
        graph_down_inv_deg_list = []
        drug_a_idx = num_gene + 1 - 1
        drug_b_idx = num_gene + 2 - 1
        for i in range(num_graph):
            # FORM ADJACENT MATRIX WITH DRUG
            adj = np.zeros((num_node, num_node))
            adj += gene_adj
            up_inv_deg = np.zeros((num_node, num_node))
            up_inv_deg += up_inv_deg_matrix
            down_inv_deg = np.zeros((num_node, num_node))
            down_inv_deg += down_inv_deg_matrix
            drug_a_degree = 0
            drug_b_degree = 0
            for row in range(num_gene):
                if xBatch[i, row, 1]:
                    adj[row, drug_a_idx] += 1
                    adj[drug_a_idx, row] += 1
                    drug_a_degree += 1
                if xBatch[i, row, 2]:
                    adj[row, drug_b_idx] += 1
                    adj[drug_b_idx, row] += 1
                    drug_b_degree += 1
            graph_adj_list.append(adj)
            # FORM DEGREE MATRIX FOR EACH NODE WITH CERTAIN ADJACENT MATRIX
            up_inv_deg[drug_a_idx, drug_a_idx] = float(1 / drug_a_degree)
            up_inv_deg[drug_b_idx, drug_b_idx] = float(1 / drug_b_degree)
            down_inv_deg[drug_a_idx, drug_a_idx] = float(1 / drug_a_degree)
            down_inv_deg[drug_b_idx, drug_b_idx] = float(1 / drug_b_degree)
            graph_up_inv_deg_list.append(up_inv_deg)
            graph_down_inv_deg_list.append(down_inv_deg)
        return graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list

    def read_label(self):
        dir_opt = self.dir_opt
        final_data_path = self.final_data_path
        RNA_seq_filename = self.RNA_seq_filename
        form_data_path = self.form_data_path
        yBatch = np.load(form_data_path + '/yBatch.npy')
        yBatch_list = [label[0] for label in list(yBatch)]
        graph_label_list = yBatch_list
        return graph_label_list


def bigraph_degree(adj):
    degree = np.zeros(len(adj))
    # CALCULATE THE SUMS ALONG ROWS AND SUM ALONG COLUMNS
    colsum = np.sum(adj, axis = 0)
    rowsum = np.sum(adj, axis = 1)
    # # LOOP THROUGH MATRIX AND ADD UP ALL DEGREE CONNECTIONS
    # for j in range(0, len(adj)):
    #     degree[j] = colsum[j] + rowsum[j]
    # degree_matrix = np.diag(degree)
    inv_colsum = (1 / colsum)
    inv_colsum[ inv_colsum == inf] = 0        
    up_inv_deg_matrix = np.diag(inv_colsum)
    inv_rowsum = (1 / rowsum)
    inv_rowsum[ inv_rowsum == inf] = 0        
    down_inv_deg_matrix = np.diag(inv_rowsum)
    return up_inv_deg_matrix, down_inv_deg_matrix


def read_train_batch(index, upper_index, args):
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    form_data_path = '.' + dir_opt + '/form_data'
    final_data_path = './data/DRUG_2'
    xTr = np.load(form_data_path + '/xTr.npy')
    yTr = np.load(form_data_path + '/yTr.npy')
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = xTr[index : upper_index, :]
    yBatch = yTr[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    np.save(form_data_path + '/xBatch.npy', xBatch)
    np.save(form_data_path + '/yBatch.npy', yBatch)
    print('READING BATCH GRAPH...')
    # cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
    # num_gene = len(cellline_gene_df)
    num_graph = xBatch.shape[0]
    num_feature = 3
    num_gene = int(xBatch.shape[1] / num_feature)
    graph_attribute_list =  ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_attribute(num_gene, num_feature)
    graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list = ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_adjacent(num_gene, num_feature, args)
    graph_label_list = ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_label()
    return graph_attribute_list, graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list, graph_label_list


def read_test_batch(index, upper_index, args):
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    form_data_path = '.' + dir_opt + '/form_data'
    final_data_path = './data/DRUG_2'
    xTe = np.load(form_data_path + '/xTe.npy')
    yTe = np.load(form_data_path + '/yTe.npy')
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = xTe[index : upper_index, :]
    yBatch = yTe[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    np.save(form_data_path + '/xBatch.npy', xBatch)
    np.save(form_data_path + '/yBatch.npy', yBatch)
    print('READING BATCH GRAPH...')
    # cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
    # num_gene = len(cellline_gene_df)
    num_graph = xBatch.shape[0]
    num_feature = 3
    num_gene = int(xBatch.shape[1] / num_feature)
    graph_attribute_list =  ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_attribute(num_gene, num_feature)
    graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list = ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_adjacent(num_gene, num_feature, args)
    graph_label_list = ReadBiGraph(dir_opt, final_data_path, RNA_seq_filename, form_data_path).read_label()
    return graph_attribute_list, graph_adj_list, graph_up_inv_deg_list, graph_down_inv_deg_list, graph_label_list

if __name__ == "__main__":
    # read_train_batch(0, 2)
    fold_split_train(5, 4)
