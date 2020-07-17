import os
import pdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

from tmain_graphsage_decoder import arg_parse, build_graphsage_model

class Analyse():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def rebuild_loss_pearson(self, path, epoch_num):
        epoch_loss_list = []
        epoch_pearson_list = []
        min_train_loss = 100
        min_train_id = 0
        for i in range(1, epoch_num + 1):
            train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
            score_list = list(train_df['Score'])
            pred_list = list(train_df['Pred Score'])
            epoch_loss = mean_squared_error(score_list, pred_list)
            epoch_loss_list.append(epoch_loss)
            epoch_pearson = train_df.corr(method = 'pearson')
            epoch_pearson_list.append(epoch_pearson['Pred Score'][0])
            if epoch_loss < min_train_loss:
                min_train_loss = epoch_loss
                min_train_id = i
        print('-------------BEST MODEL ID:' + str(min_train_id) + '-------------')
        print('BEST MODEL TRAIN LOSS: ', min_train_loss)
        print('BEST MODEL PEARSON CORR: ', epoch_pearson_list[min_train_id - 1])
        print('\n-------------EPOCH TRAINING PEARSON CORRELATION LIST: -------------')
        print(epoch_pearson_list)
        print('\n-------------EPOCH TRAINING MSE LOSS LIST: -------------')
        print(epoch_loss_list)
        epoch_pearson_array = np.array(epoch_pearson_list)
        epoch_loss_array = np.array(epoch_loss_list)
        np.save(path + '/pearson.npy', epoch_pearson_array)
        np.save(path + '/loss.npy', epoch_loss_array)

    def plot_loss_pearson(self, path, epoch_num):
        epoch_pearson_array = np.load(path + '/pearson.npy')
        epoch_loss_array = np.load(path + '/loss.npy')
        x = range(1, epoch_num + 1)
        plt.figure(1)
        plt.title('Training Loss and Pearson Correlation in ' + str(epoch_num) + ' Epochs') 
        plt.xlabel('Train Epochs') 
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x, epoch_loss_array) 
        plt.subplot(212)
        plt.plot(x, epoch_pearson_array)
        plt.show()

    def reform_weight_adj(self, RNA_seq_filename, model):
        dir_opt = self.dir_opt
        # WEIGHT PARAMETETS IN MODEL, ALSO (Src -> Dest)
        if model:
            print('LOADING WEIGHT PARAMETERS FROM SAVED MODEL...')
            first_conv_weight = model.conv_first.weight_adj.cpu().data.numpy()
            block_conv_weight = model.conv_block[0].weight_adj.cpu().data.numpy()
            last_conv_weight = model.conv_last.weight_adj.cpu().data.numpy()
            if os.path.exists('.' + dir_opt + '/analyse_data') == False:
                os.mkdir('.' + dir_opt + '/analyse_data')
            np.save('.' + dir_opt + '/analyse_data/first_conv_weight.npy', first_conv_weight)
            np.save('.' + dir_opt + '/analyse_data/block_conv_weight.npy', block_conv_weight)
            np.save('.' + dir_opt + '/analyse_data/last_conv_weight.npy', last_conv_weight)
            # print(first_conv_weight)
            # print(block_conv_weight)
            # print(last_conv_weight)
        else:
            print('LOADING WEIGHT FROM SAVE NUMPY FILES...')
            first_conv_weight = np.load('.' + dir_opt + '/analyse_data/first_conv_weight.npy')
            block_conv_weight = np.load('.' + dir_opt + '/analyse_data/block_conv_weight.npy')
            last_conv_weight = np.load('.' + dir_opt + '/analyse_data/last_conv_weight.npy')

        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        print(cellline_gene_num_dict)

        # GENE-GENE ADJACENT MATRIX (Src -> Dest)
        form_data_path = '.' + dir_opt + '/form_data'
        cellline_gene_num_df = pd.read_csv(form_data_path + '/gene_connection_num.txt')
        src_gene_list = list(cellline_gene_num_df['src'])
        dest_gene_list = list(cellline_gene_num_df['dest'])
        num_gene = len(cellline_gene_df)
        num_node = num_gene + 2
        gene_adj = np.zeros((num_node, num_node))
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            gene_adj[row_idx, col_idx] = 1
        
        # WEIGHTING [Absolute Value] ADJACENT GENE-GENE MATRICES (Src -> Dest)
        first_conv_weight_adj = np.multiply(gene_adj, np.absolute(first_conv_weight))
        block_conv_weight_adj = np.multiply(gene_adj, np.absolute(block_conv_weight))
        last_conv_weight_adj = np.multiply(gene_adj, np.absolute(last_conv_weight))
        print(first_conv_weight_adj)
        
        # CONVERT ADJACENT MATRICES TO EDGES, GENE ID STARTS WITH 1
        src_gene_name_list = []
        dest_gene_name_list = []
        weight_src_gene_list = []
        weight_dest_gene_list = []
        weight_edge_list = []
        count = 0
        for row in range(num_gene):
            for col in range(num_gene):
                if gene_adj[row, col]:
                    src_gene_name_list.append(cellline_gene_num_dict[row + 1])
                    dest_gene_name_list.append(cellline_gene_num_dict[col + 1])
                    weight_src_gene_list.append(row + 1)
                    weight_dest_gene_list.append(col + 1)
                    # ########ONLY USE FIRST CONV WEIGHT########
                    weight_edge = first_conv_weight_adj[row, col]
                    weight_edge_list.append(weight_edge)
        weight_src_dest = {'src': weight_src_gene_list, 'src_name': src_gene_name_list, \
                        'dest': weight_dest_gene_list, 'dest_name': dest_gene_name_list, \
                        'weight': weight_edge_list}
        gene_weight_edge_df = pd.DataFrame(weight_src_dest)
        print(gene_weight_edge_df)
        gene_weight_edge_df.to_csv('.' + dir_opt + '/analyse_data/gene_weight_edge.csv', index = False, header = True)

        # CONVERT ADJACENT MATRICES WITH GENE OUT/IN DEGREE
        gene_list = []
        gene_name_list = []
        gene_outdeg_list = []
        gene_indeg_list = []
        gene_degree_list = []
        for idx in range(num_gene):
            gene_list.append(idx + 1)
            gene_name_list.append(cellline_gene_num_dict[idx + 1])
            # ########ONLY USE FIRST CONV WEIGHT########
            gene_outdeg = np.sum(first_conv_weight_adj[idx, :])
            gene_outdeg_list.append(gene_outdeg)
            # ########ONLY USE FIRST CONV WEIGHT########
            gene_indeg = np.sum(first_conv_weight_adj[:, idx])
            gene_indeg_list.append(gene_indeg)
            gene_degree = gene_outdeg + gene_indeg
            gene_degree_list.append(gene_degree)
        weight_gene_degree = {'gene_idx': gene_list, 'gene_name': gene_name_list, 'out_degree': gene_outdeg_list,\
                        'in_degree': gene_indeg_list, 'degree': gene_degree_list}
        gene_weight_degree_df = pd.DataFrame(weight_gene_degree)
        print(gene_weight_degree_df)
        gene_weight_degree_df.to_csv('.' + dir_opt + '/analyse_data/gene_weight_degree.csv', index = False, header = True)



if __name__ == "__main__":
    # BASICAL PARAMETERS IN FILES
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    path = '.' + dir_opt + '/result/epoch_100_2.2'

    # ANALYSE [MSE_LOSS/PEARSON CORRELATION] FROM RECORDED FILES
    epoch_num = 100
    Analyse(dir_opt).rebuild_loss_pearson(path, epoch_num)
    # Analyse(dir_opt).plot_loss_pearson(path, epoch_num)

    # # REBUILD MODEL AND ANALYSIS PARAMTERS
    # prog_args = arg_parse()
    # load_model = False
    # if load_model == True:
    #     model = build_graphsage_model(prog_args)
    #     model.load_state_dict(torch.load('./datainfo2/result/epoch_100_2d/best_train_model.pth'))
    # else:
    #     model = 0
    # Analyse(dir_opt).reform_weight_adj(RNA_seq_filename, model)