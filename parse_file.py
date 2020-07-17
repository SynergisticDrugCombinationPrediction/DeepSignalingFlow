import os
import pdb
import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.model_selection import train_test_split

from load_data import LoadData



# USE MAP DICT TO READ INPUT FROM DEEPLEARNING
# drug -[drug_map]-> drug_name(drug_i, drug_j)
# celllinename -[celllinemap]-> cellline_name
#                 -> gene_name -[drug_map][drug_target]-> (RNA, drug_i, drug_j)
class ParseFile():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    # FIND THE DUPLICATE ROWS[Drug A, Drug B, Cell Line Name] THEN AVERAGE SCORE
    def input_condense(self):
        dir_opt = self.dir_opt
        dl_input_df = pd.read_csv('.' + dir_opt + '/init_data/DeepLearningInput.csv')
        dl_input_df = dl_input_df.groupby(['Drug A', 'Drug B', 'Cell Line Name']).agg({'Score':'mean'}).reset_index()
        dl_input_df.to_csv('.' + dir_opt + '/mid_data/DeepLearningInput.txt', index = False, header = True)

    # REMOVE INPUT ROWS WITH NO MAPPED DRUG NAME (48953 POINTS INPUT)
    def input_drug_condense(self):
        dir_opt = self.dir_opt
        dl_input_df = pd.read_table('.' + dir_opt + '/mid_data/DeepLearningInput.txt', delimiter = ',')
        drug_map_dict = ParseFile(dir_opt).drug_map_dict()
        deletion_list = []
        for row in dl_input_df.itertuples():
            if pd.isna(drug_map_dict[row[1]]) or pd.isna(drug_map_dict[row[2]]):
                deletion_list.append(row[0])
        mid_dl_input_df = dl_input_df.drop(dl_input_df.index[deletion_list]).reset_index(drop = True)
        mid_dl_input_df.to_csv('.' + dir_opt + '/mid_data/MidDeepLearningInput.txt', index = False, header = True)
    
    # REMOVE INPUT ROWS WITH NO CORRESPONDING CELLLINE NAME ([, 37355] POINTS INPUT)
    def input_cellline_condense(self, RNA_seq_filename):
        dir_opt = self.dir_opt
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        cellline_name_list =  list(cellline_gene_df.columns[2:])
        mid_dl_input_df = pd.read_table('.' + dir_opt + '/mid_data/MidDeepLearningInput.txt', delimiter = ',')
        cellline_map_dict = ParseFile(dir_opt).cellline_map_dict()
        deletion_list = []
        for row in mid_dl_input_df.itertuples():
            if cellline_map_dict[row[3]] not in cellline_name_list:
                deletion_list.append(row[0])
        final_dl_input_df = mid_dl_input_df.drop(mid_dl_input_df.index[deletion_list]).reset_index(drop = True)
        final_dl_input_df.to_csv('.' + dir_opt + '/mid_data/FinalDeepLearningInput.txt', index = False, header = True)
    
    # REMOVE INPUT ROWS WITH ALL ZEROS ON DRUG TARGET GENE CONNECTION,
    # WHICH MEANS NO INPUT POINT(VECTOR) HAS ALL ZERO ON 1634 GENES
    def input_drug_gene_condense(self, RNA_seq_filename):
        dir_opt = self.dir_opt
        deletion_list = []
        final_dl_input_df = pd.read_table('.' + dir_opt + '/mid_data/FinalDeepLearningInput.txt', delimiter = ',')
        drug_map_dict, cellline_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt, RNA_seq_filename).pre_load_dict()
        target_index_list = gene_target_num_dict.values()
        drug_target_matrix = np.load('.' + dir_opt + '/filtered_data/drug_target_matrix.npy')
        for row in final_dl_input_df.itertuples():
            drug_a = drug_map_dict[row[1]]
            drug_b = drug_map_dict[row[2]]
            cellline_name = cellline_map_dict[row[3]]
            # DRUG_A AND 1634 TARGET GENES
            drug_a_target_list = []
            drug_index = drug_dict[drug_a]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_a_target_list.append(effect)
            # DRUG_B AND 1634 TARGET GENES
            drug_b_target_list = []
            drug_index = drug_dict[drug_b]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_b_target_list.append(effect)
            if all([a == 0 for a in drug_a_target_list]) or all([b == 0 for b in drug_b_target_list]): 
                deletion_list.append(row[0])
        zero_final_dl_input_df = final_dl_input_df.drop(final_dl_input_df.index[deletion_list]).reset_index(drop = True)
        zero_final_dl_input_df.to_csv('.' + dir_opt + '/filtered_data/ZeroFinalDeepLearningInput.txt', index = False, header = True)
        # print(zero_final_dl_input_df)

    # RANDOMIZE THE DL INPUT
    def input_random_condense(self):
        dir_opt = self.dir_opt
        zero_final_dl_input_df = pd.read_table('.' + dir_opt + '/filtered_data/ZeroFinalDeepLearningInput.txt', delimiter = ',')
        random_final_dl_input_df = zero_final_dl_input_df.sample(frac = 1)
        random_final_dl_input_df.to_csv('.' + dir_opt + '/filtered_data/RandomFinalDeepLearningInput.txt', index = False, header = True)
        print(random_final_dl_input_df)

    # CALCULATE NUMBER OF UNIQUE DRUG IN RANDOMFINAL_INPUT
    def random_final_drug_count(self):
        dir_opt = self.dir_opt
        random_final_dl_input_df = pd.read_table('.' + dir_opt + '/filtered_data/RandomFinalDeepLearningInput.txt', delimiter = ',')
        random_final_drug_list = []
        for drug in random_final_dl_input_df['Drug A']:
            if drug not in random_final_drug_list:
                random_final_drug_list.append(drug)
        for drug in random_final_dl_input_df['Drug B']:
            if drug not in random_final_drug_list:
                random_final_drug_list.append(drug)
        random_final_drug_list = sorted(random_final_drug_list)
        print(random_final_drug_list)
        print(len(random_final_drug_list))

    # # SPLIT DEEP LEARNING INPUT INTO TRAINING AND TEST
    def split_k_fold(self, k, place_num):
        dir_opt = self.dir_opt
        dir_opt = self.dir_opt
        random_final_dl_input_df = pd.read_table('.' + dir_opt + '/filtered_data/RandomFinalDeepLearningInput.txt', delimiter = ',')
        print(random_final_dl_input_df)
        num_points = random_final_dl_input_df.shape[0]

        num_div = int(num_points / k)
        num_div_list = [i * num_div for i in range(0, k)]
        num_div_list.append(num_points)
        low_idx = num_div_list[place_num - 1]
        high_idx = num_div_list[place_num]
        print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
        train_input_df = random_final_dl_input_df.drop(random_final_dl_input_df.index[low_idx : high_idx])
        print(train_input_df)
        test_input_df = random_final_dl_input_df[low_idx : high_idx]
        print(test_input_df)
        train_input_df.to_csv('.' + dir_opt + '/filtered_data/TrainingInput.txt', index = False, header = True)
        test_input_df.to_csv('.' + dir_opt + '/filtered_data/TestInput.txt', index = False, header = True)



    # FIND UNIQUE DRUG NAME FROM DATAFRAME AND MAP 
    def drug_map(self):
        dir_opt = self.dir_opt
        dl_input_df = pd.read_table('.' + dir_opt + '/mid_data/DeepLearningInput.txt', delimiter = ',')
        drug_target_df = pd.read_table('.' + dir_opt + '/init_data/drug_tar_drugBank_all.txt')
        drug_list = []
        for drug in dl_input_df['Drug A']:
            if drug not in drug_list:
                drug_list.append(drug)
        for drug in dl_input_df['Drug B']:
            if drug not in drug_list:
                drug_list.append(drug)
        drug_list = sorted(drug_list)
        drug_df = pd.DataFrame(data = drug_list, columns = ['Drug Name'])
        drug_df.to_csv('.' + dir_opt + '/init_data/input_drug_name.txt', index = False, header = True)
        mapped_drug_list = []
        for drug in drug_target_df['Drug']:
            if drug not in mapped_drug_list:
                mapped_drug_list.append(drug)
        mapped_drug_list = sorted(mapped_drug_list)
        mapped_drug_df = pd.DataFrame(data = mapped_drug_list, columns = ['Mapped Drug Name'])
        mapped_drug_df.to_csv('.' + dir_opt + '/init_data/mapped_drug_name.txt', index = False, header = True)
        # LEFT JOIN TWO DATAFRAME
        drug_map_df = pd.merge(drug_df, mapped_drug_df, how='left', left_on = 'Drug Name', right_on = 'Mapped Drug Name')
        drug_map_df.to_csv('.' + dir_opt + '/init_data/drug_map.csv', index = False, header = True)
        # AFTER AUTO MAP -> MANUAL MAP
    
    # FROM MANUAL MAP TO DRUG MAP DICT
    def drug_map_dict(self):
        dir_opt = self.dir_opt
        drug_map_df = pd.read_csv('.' + dir_opt + '/mid_data/drug_map.csv')
        drug_map_dict = {}
        for row in drug_map_df.itertuples():
            drug_map_dict[row[1]] = row[2]
        if os.path.exists('.' + dir_opt + '/filtered_data') == False:
            os.mkdir('.' + dir_opt + '/filtered_data')
        np.save('.' + dir_opt + '/filtered_data/drug_map_dict.npy', drug_map_dict)
        return drug_map_dict

    # FORM ADAJACENT MATRIX (DRUG x TARGET) (LIST -> SORTED -> DICT -> MATRIX) (ALL 5435 DRUGS <-> ALL 2775 GENES)
    def drug_target(self):
        dir_opt = self.dir_opt
        drug_target_df = pd.read_table('.' + dir_opt + '/init_data/drug_tar_drugBank_all.txt')
        # GET UNIQUE SORTED DRUGLIST AND TARGET(GENE) LIST
        drug_list = []
        for drug in drug_target_df['Drug']:
            if drug not in drug_list:
                drug_list.append(drug)
        drug_list = sorted(drug_list)
        target_list = []
        for target in drug_target_df['Target']:
            if target not in target_list:
                target_list.append(target)
        target_list = sorted(target_list)
        # CONVERT THE SORTED LIST TO DICT WITH VALUE OF INDEX
        drug_dict = {drug_list[i] : i for i in range((len(drug_list)))} 
        drug_num_dict = {i : drug_list[i] for i in range((len(drug_list)))} 
        target_dict = {target_list[i] : i for i in range(len(target_list))}
        target_num_dict = {i : target_list[i] for i in range(len(target_list))}
        # ITERATE THE DATAFRAME TO DEFINE CONNETIONS BETWEEN DRUG AND TARGET(GENE)
        drug_target_matrix = np.zeros((len(drug_list), len(target_list))).astype(int)
        for index, drug_target in drug_target_df.iterrows():
            # BUILD ADJACENT MATRIX
            drug_target_matrix[drug_dict[drug_target['Drug']], target_dict[drug_target['Target']]] = 1
        drug_target_matrix = drug_target_matrix.astype(int)
        np.save('.' + dir_opt + '/filtered_data/drug_target_matrix.npy', drug_target_matrix)
        # np.savetxt("drug_target_matrix.csv", drug_target_matrix, delimiter=',')
        # x, y = drug_target_matrix.shape
        # for i in range(x):
        #     # FIND DRUG TARGET OVER 100 GENES
        #     row = drug_target_matrix[i, :]
        #     if len(row[row>=1]) >= 100: print(drug_num_dict[i])
        np.save('.' + dir_opt + '/filtered_data/drug_dict.npy', drug_dict)
        np.save('.' + dir_opt + '/filtered_data/drug_num_dict.npy', drug_num_dict)
        np.save('.' + dir_opt + '/filtered_data/target_dict.npy', target_dict)
        np.save('.' + dir_opt + '/filtered_data/target_num_dict.npy', target_num_dict)
        return drug_dict, drug_num_dict, target_dict, target_num_dict

    # FROM MANUAL CELLLINE NAME MAP TO DICT
    def cellline_map_dict(self):
        dir_opt = self.dir_opt
        cellline_name_df = pd.read_table('.' + dir_opt + '/init_data/nci60-ccle_cell_name_map1.txt')
        cellline_map_dict = {}
        for row in cellline_name_df.itertuples():
            cellline_map_dict[row[1]] = row[2]
        np.save('.' + dir_opt + '/filtered_data/cellline_map_dict.npy', cellline_map_dict)
        return cellline_map_dict

    # FILTER DUPLICATED AND SPARSE GENES (FINALLY [1118, 1684] GENES)
    def filter_cellline_gene(self, RNA_seq_filename):
        dir_opt = self.dir_opt
        cellline_gene_df = pd.read_table('.' + dir_opt + '/init_data/' + RNA_seq_filename + '.txt')
        cellline_gene_df = cellline_gene_df.drop_duplicates(subset = ['geneSymbol'], 
                    keep = 'first').sort_values(by = ['geneSymbol']).reset_index(drop = True)
        threshold = int((len(cellline_gene_df.columns) - 3) / 3)
        deletion_list = []
        for row in cellline_gene_df.itertuples():
            if list(row[3:]).count(0) > threshold: 
                deletion_list.append(row[0])
        cellline_gene_df = cellline_gene_df.drop(cellline_gene_df.index[deletion_list]).reset_index(drop = True)     
        cellline_gene_df.to_csv('.' + dir_opt + '/mid_data/' + RNA_seq_filename + '.csv', index = False, header = True)
        print(cellline_gene_df)

    # FILTER GENES NOT EXIST IN EDGES FILE(FINALLY [, 1634] GENES)
    # AND FORM TUPLES GENE CONNECTION AS EDGES
    def filter_form_edge_cellline_gene(self, RNA_seq_filename, gene_filename, form_data_path):
        dir_opt = self.dir_opt
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/mid_data/' + RNA_seq_filename +'.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        # DELETE GENES NOT EXIST IN [cellline_gene_df] WRT FILE [Selected_Kegg_Pathways_edges_1.txt]
        gene_connection_df = pd.read_table('.' + dir_opt + '/init_data/' + gene_filename + '.txt')
        gene_connection_df = gene_connection_df.drop(columns = ['src_type', 'dest_type', 'direction', 'type'])
        gene_connection_deletion_list = []
        for row in gene_connection_df.itertuples():
            if row[1] not in cellline_gene_list or row[2] not in cellline_gene_list:
                gene_connection_deletion_list.append(row[0])
        gene_connection_df = gene_connection_df.drop(gene_connection_df.index[gene_connection_deletion_list]).reset_index(drop = True)
        # SORT DELETED [gene_connection_df] IN ALPHA-BETA ORDER
        gene_connection_df = gene_connection_df.sort_values(by = ['src', 'dest']).reset_index(drop = True)
        # FETCH ALL UNIQUE GENE IN [gene_connection_df] [1634 genes]
        gene_connection_list = []
        for row in gene_connection_df.itertuples():
            if row[1] not in gene_connection_list:
                gene_connection_list.append(row[1])
            if row[2] not in gene_connection_list:
                gene_connection_list.append(row[2])
        print(len(gene_connection_list))
        # DELETE GENES NOT EXIST IN [gene_connection_list] WRT [cellline_gene_gf]
        cellline_gene_deletion_list = []
        for row in cellline_gene_df.itertuples():
            if row[2] not in gene_connection_list:
                cellline_gene_deletion_list.append(row[0])
        print(len(cellline_gene_deletion_list))
        cellline_gene_df = cellline_gene_df.drop(cellline_gene_df.index[cellline_gene_deletion_list]).reset_index(drop = True)     
        cellline_gene_df.to_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv', index = False, header = True)
        print(cellline_gene_df)
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1 INSTEAD OF 0 !!!
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_dict = {cellline_gene_list[i - 1] : i for i in range(1, len(cellline_gene_list) + 1)}
        # FORM TUPLES GENE CONNECTION ACCORDING TO NUM INDEX OF GENE IN [cellline_gene_dict]
        src_gene_list = []
        dest_gene_list = []
        for row in gene_connection_df.itertuples():
            src_gene_list.append(cellline_gene_dict[row[1]])
            dest_gene_list.append(cellline_gene_dict[row[2]])
        src_dest = {'src': src_gene_list, 'dest': dest_gene_list}
        gene_connection_num_df = pd.DataFrame(src_dest)
        if os.path.exists(form_data_path) == False:
            os.mkdir(form_data_path)
        gene_connection_num_df.to_csv(form_data_path + '/gene_connection_num.txt', index = False, header = True)


    # BUILD GENES MAP BETWEEN [cellline_gene_df] AND [target_dict] 
    # [CCLE GENES : DRUG_TAR GENES]  KEY : VALUE  (1634 <-MAP-> 2775)
    def gene_target_num_dict(self, RNA_seq_filename):
        dir_opt = self.dir_opt
        drug_dict, drug_num_dict, target_dict, target_num_dict = ParseFile(dir_opt).drug_target()
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename +'.csv')
        # print(target_dict)
        gene_target_num_dict = {}
        for row in cellline_gene_df.itertuples():
            if row[2] not in target_dict.keys(): 
                map_index = -1
            else:
                map_index = target_dict[row[2]]
            gene_target_num_dict[row[0]] = map_index
        np.save('.' + dir_opt + '/filtered_data/gene_target_num_dict.npy', gene_target_num_dict)
        return gene_target_num_dict

    # FORM ADAJACENT MATRIX (GENE x PATHWAY) (LIST -> SORTED -> DICT -> MATRIX) (ALL 1298 GENES <-> 16 PATHWAYS)
    def gene_pathway(self, pathway_filename):
        dir_opt = self.dir_opt
        gene_pathway_df = pd.read_table('.' + dir_opt + '/init_data/' + pathway_filename + '.txt')
        gene_list = sorted(list(gene_pathway_df['AllGenes']))
        gene_pathway_df = gene_pathway_df.drop(['AllGenes'], axis = 1).sort_index(axis = 1)
        pathway_list = list(gene_pathway_df.columns)
        # CONVERT SORTED LIST TO DICT WITH INDEX
        gene_dict = {gene_list[i] : i for i in range(len(gene_list))}
        gene_num_dict = {i : gene_list[i] for i in range(len(gene_list))}
        pathway_dict = {pathway_list[i] : i for i in range(len(pathway_list))}
        pathway_num_dict = {i : pathway_list[i] for i in range(len(pathway_list))}
        # ITERATE THE DATAFRAME TO DEFINE CONNETIONS BETWEEN GENES AND PATHWAYS
        gene_pathway_matrix = np.zeros((len(gene_list), len(pathway_list))).astype(int)
        print(gene_pathway_matrix.shape)
        for gene_row in gene_pathway_df.itertuples():
            pathway_index = 0
            for gene in gene_row[1:]:
                if gene != 'test':
                    gene_pathway_matrix[gene_dict[gene], pathway_index] = 1
                pathway_index += 1
        np.save('.' + dir_opt + '/filtered_data/gene_pathway_matrix.npy', gene_pathway_matrix)
        np.save('.' + dir_opt + '/filtered_data/gene_dict.npy', gene_dict)
        np.save('.' + dir_opt + '/filtered_data/gene_num_dict.npy', gene_num_dict)
        np.save('.' + dir_opt + '/filtered_data/pathway_dict.npy', pathway_dict)
        np.save('.' + dir_opt + '/filtered_data/pathway_num_dict.npy', pathway_num_dict)
        return gene_dict, gene_num_dict, pathway_dict, pathway_num_dict


def pre_manual():
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    ParseFile(dir_opt).input_condense()
    ParseFile(dir_opt).drug_map()
    # AFTER GET [/init_data/drug_map.csv] WITH AUTO MAP -> MANUAL MAP

def pre_parse():
    dir_opt = '/datainfo2'
    # STABLE DICTIONARY NOT CHANGE WITH FILES
    ParseFile(dir_opt).drug_map_dict()
    ParseFile(dir_opt).drug_target()
    ParseFile(dir_opt).cellline_map_dict()
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    ParseFile(dir_opt).filter_cellline_gene(RNA_seq_filename)
    # FILTER GENES NOT IN CELLLINE FOR EDGES
    # FILTER GENES NOT IN EDGES FOR CELLLINE
    gene_filename = 'Selected_Kegg_Pathways_edges_1'
    form_data_path = '.' + dir_opt + '/form_data'
    ParseFile(dir_opt).filter_form_edge_cellline_gene(RNA_seq_filename, gene_filename, form_data_path)
    ParseFile(dir_opt).gene_target_num_dict(RNA_seq_filename)
    pathway_filename = 'Selected_Kegg_Pathways2'
    ParseFile(dir_opt).gene_pathway(pathway_filename)

def pre_input():
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    ParseFile(dir_opt).input_condense()
    ParseFile(dir_opt).input_drug_condense()
    ParseFile(dir_opt).input_cellline_condense(RNA_seq_filename)
    ParseFile(dir_opt).input_drug_gene_condense(RNA_seq_filename)
    # ParseFile(dir_opt).random_final_drug_count()

def k_fold_split(random_mode, k, place_num):
    dir_opt = '/datainfo2'
    if random_mode == True:
        ParseFile(dir_opt).input_random_condense()
        ParseFile(dir_opt).random_final_drug_count()
    ParseFile(dir_opt).split_k_fold(k, place_num)

def pre_form():
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    form_data_path = '.' + dir_opt + '/form_data'
    batch_size = 256
    xTr, yTr, xTe, yTe, x, y = LoadData(dir_opt, RNA_seq_filename).load_all(batch_size, form_data_path)
    

if __name__ == "__main__":
    pre_parse()
    pre_input()

    # DOING K-FOLD VALIDATION IN 100% DATASET
    random_mode = False
    k = 5
    place_num = 1
    k_fold_split(random_mode, k, place_num)
    # FORM NUMPY FILES TO BE LOADED
    pre_form()


    