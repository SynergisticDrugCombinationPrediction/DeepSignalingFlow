import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self, dir_opt, RNA_seq_filename):
        self.dir_opt = dir_opt
        self.RNA_seq_filename = RNA_seq_filename

    def pre_load_dict(self):
        dir_opt = self.dir_opt
        drug_map_dict = np.load('.' + dir_opt + '/filtered_data/drug_map_dict.npy', allow_pickle='TRUE').item()
        cellline_map_dict = np.load('.' + dir_opt + '/filtered_data/cellline_map_dict.npy', allow_pickle='TRUE').item()
        drug_dict = np.load('.' + dir_opt + '/filtered_data/drug_dict.npy', allow_pickle='TRUE').item()
        drug_num_dict = np.load('.' + dir_opt + '/filtered_data/drug_num_dict.npy', allow_pickle='TRUE').item()
        target_dict = np.load('.' + dir_opt + '/filtered_data/target_dict.npy', allow_pickle='TRUE').item()
        target_num_dict = np.load('.' + dir_opt + '/filtered_data/target_num_dict.npy', allow_pickle='TRUE').item()
        gene_target_num_dict = np.load('.' + dir_opt + '/filtered_data/gene_target_num_dict.npy', allow_pickle='TRUE').item()
        return drug_map_dict, cellline_map_dict, drug_dict, gene_target_num_dict

    def pre_load_train(self):
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        train_input_df = pd.read_table('.' + dir_opt + '/filtered_data/TrainingInput.txt', delimiter = ',')
        input_num, input_dim = train_input_df.shape
        num_feature = input_dim - 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        num_gene, num_cellline = cellline_gene_df.shape
        gene_pathway_matrix = np.load('.' + dir_opt + '/filtered_data/gene_pathway_matrix.npy')
        all_gene, num_pathway =  gene_pathway_matrix.shape
        return train_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway

    def load_train(self, index, upper_index):
        # LOAD 80 PERCENT TRAINING DATA
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        train_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_train()
        drug_map_dict, cellline_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt, RNA_seq_filename).pre_load_dict()
        target_index_list = gene_target_num_dict.values()
        drug_target_matrix = np.load('.' + dir_opt + '/filtered_data/drug_target_matrix.npy')
        # COMBINE A BATCH SIZE AS [xTr_batch, yTr_batch]
        print('-----' + str(index) + ' to ' + str(upper_index) + '-----')
        tmp_batch_size = 0
        y_input_list = []
        xTr_batch = np.zeros((1, (num_feature * num_gene)))
        for row in train_input_df.iloc[index : upper_index].itertuples():
            tmp_batch_size += 1
            drug_a = drug_map_dict[row[1]]
            drug_b = drug_map_dict[row[2]]
            cellline_name = cellline_map_dict[row[3]]
            # y = int(row[4]>0) BINARY CLASSIFICATION
            y = row[4]
            # DRUG_A AND 1130 TARGET GENES
            drug_a_target_list = []
            drug_index = drug_dict[drug_a]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_a_target_list.append(effect)
            # DRUG_B AND 1130 TARGET GENES
            drug_b_target_list = []
            drug_index = drug_dict[drug_b]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_b_target_list.append(effect)
            # GENE RNA SEQUENCE
            gene_rna_list = list(cellline_gene_df[cellline_name])
            # COMBINE RNA, DRUG_A_TARGET, DRUG_B_TARGET
            x_input_list = []
            for i in range(num_gene):
                x_input_list.append(gene_rna_list[i])
                x_input_list.append(drug_a_target_list[i])
                x_input_list.append(drug_b_target_list[i])
            x_input = np.array(x_input_list)
            xTr_batch = np.vstack((xTr_batch, x_input))
            # COMBINE SCORE LIST
            y_input_list.append(y)
        xTr_batch = np.delete(xTr_batch, 0, axis = 0)
        yTr_batch = np.array(y_input_list).reshape(tmp_batch_size, 1)
        print(xTr_batch.shape)
        print(yTr_batch.shape)
        return xTr_batch, yTr_batch

    def pre_load_test(self):
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        test_input_df = pd.read_table('.' + dir_opt + '/filtered_data/TestInput.txt', delimiter = ',')
        input_num, input_dim = test_input_df.shape
        num_feature = input_dim - 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        num_gene, num_cellline = cellline_gene_df.shape
        gene_pathway_matrix = np.load('.' + dir_opt + '/filtered_data/gene_pathway_matrix.npy')
        all_gene, num_pathway =  gene_pathway_matrix.shape
        return test_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway

    def load_test(self):
        # LOAD 20 PERCENT TEST DATA
        print('LOADING 20 PERCENT TEST DATA...')
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        test_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_test()
        drug_map_dict, cellline_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt, RNA_seq_filename).pre_load_dict()
        target_index_list = gene_target_num_dict.values()
        drug_target_matrix = np.load('.' + dir_opt + '/filtered_data/drug_target_matrix.npy')
        # COMBINE ALL AS [xTe, yTe]
        test_size = 0
        y_input_list = []
        xTe = np.zeros((1, (num_feature * num_gene)))
        for row in test_input_df.itertuples():
            test_size += 1
            drug_a = drug_map_dict[row[1]]
            drug_b = drug_map_dict[row[2]]
            cellline_name = cellline_map_dict[row[3]]
            # y = int(row[4]>0) BINARY CLASSIFICATION
            y = row[4]
            # DRUG_A AND 1130 TARGET GENES
            drug_a_target_list = []
            drug_index = drug_dict[drug_a]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_a_target_list.append(effect)
            # DRUG_B AND 1130 TARGET GENES
            drug_b_target_list = []
            drug_index = drug_dict[drug_b]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_b_target_list.append(effect)
            # GENE RNA SEQUENCE
            gene_rna_list = list(cellline_gene_df[cellline_name])
            # COMBINE RNA, DRUG_A_TARGET, DRUG_B_TARGET
            x_input_list = []
            for i in range(num_gene):
                x_input_list.append(gene_rna_list[i])
                x_input_list.append(drug_a_target_list[i])
                x_input_list.append(drug_b_target_list[i])
            x_input = np.array(x_input_list)
            xTe = np.vstack((xTe, x_input))
            # COMBINE SCORE LIST
            y_input_list.append(y)
        xTe = np.delete(xTe, 0, axis = 0)
        yTe = np.array(y_input_list).reshape(test_size, 1)
        print(xTe.shape)
        print(yTe.shape)
        return xTe, yTe

    def pre_load_all(self):
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        all_input_df = pd.read_table('.' + dir_opt + '/filtered_data/ZeroFinalDeepLearningInput.txt', delimiter = ',')
        input_num, input_dim = all_input_df.shape
        num_feature = input_dim - 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        num_gene, num_cellline = cellline_gene_df.shape
        gene_pathway_matrix = np.load('.' + dir_opt + '/filtered_data/gene_pathway_matrix.npy')
        all_gene, num_pathway =  gene_pathway_matrix.shape
        return all_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway

    def load_all(self, batch_size, form_data_path):
        # LOAD 100 PERCENT DATA
        print('LOADING ALL DATA...')
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        # FIRST LOAD 80 PERCENT TRAINING DATA
        batch_size = 256
        train_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_train()
        xTr = np.zeros((1, num_feature * num_gene))
        yTr = np.zeros((1, 1))
        for index in range(0, input_num, batch_size):
            if (index + batch_size) < input_num:
                upper_index = index + batch_size
            else:
                upper_index = input_num
            xTr_batch, yTr_batch = LoadData(dir_opt, RNA_seq_filename).load_train(index, upper_index)
            xTr = np.vstack((xTr, xTr_batch))
            yTr = np.vstack((yTr, yTr_batch))
        xTr = np.delete(xTr, 0, axis = 0)
        yTr = np.delete(yTr, 0, axis = 0)
        print('-------TRAINING DATA SHAPE-------')
        print(xTr.shape)
        print(yTr.shape)
        np.save(form_data_path + '/xTr.npy', xTr)
        np.save(form_data_path + '/yTr.npy', yTr)
        # THEN LOAD 20 PERCENT TEST DATA
        print('-------TEST DATA SHAPE-----------')
        xTe, yTe = LoadData(dir_opt, RNA_seq_filename).load_test()
        np.save(form_data_path + '/xTe.npy', xTe)
        np.save(form_data_path + '/yTe.npy', yTe)
        # COMBINE TRAINING AND TEST DATA
        print('-------ALL DATA SHAPE------------')
        x = np.vstack((xTr, xTe))
        y = np.vstack((yTr, yTe))
        print(x.shape)
        print(y.shape)
        np.save(form_data_path + '/x.npy', x)
        np.save(form_data_path + '/y.npy', y)
        return xTr, yTr, xTe, yTe, x, y