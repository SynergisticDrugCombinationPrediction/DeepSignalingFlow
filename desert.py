###############
# bigraphsage_dec_analysis.py
###############

# # CONVERT ADJACENT MATRICES TO EDGES, GENE ID STARTS WITH 1 (BIND WITH UP/DOWN)
# bind_src_gene_name_list = []
# bind_dest_gene_name_list = []
# bind_src_gene_num_list = []
# bind_dest_gene_num_list = []
# bind_weight_edge_list = []
# for row in range(num_gene):
#     for col in range(num_gene):
#         if gene_adj[row, col]:
#             # RECORD FOR [src -> dest]
#             bind_src_gene_name_list.append(cellline_gene_num_dict[row + 1])
#             bind_dest_gene_name_list.append(cellline_gene_num_dict[col + 1])
#             bind_src_gene_num_list.append(row + 1)
#             bind_dest_gene_num_list.append(col + 1)
#             up_weight_edge = conv_up_weight_adj_bind[row, col]
#             bind_weight_edge_list.append(up_weight_edge)
#             # RECORD FOR [dest -> src]
#             bind_src_gene_name_list.append(cellline_gene_num_dict[col + 1])
#             bind_dest_gene_name_list.append(cellline_gene_num_dict[row + 1])
#             bind_src_gene_num_list.append(col + 1)
#             bind_dest_gene_num_list.append(row + 1)
#             down_weight_edge = conv_down_weight_adj_bind[row, col]
#             bind_weight_edge_list.append(down_weight_edge)
# bind_weight_src_dest = {'src': bind_src_gene_num_list, 'src_name': bind_src_gene_name_list, \
#                 'dest': bind_dest_gene_num_list, 'dest_name': bind_dest_gene_name_list, \
#                 'weight': bind_weight_edge_list}
# gene_bind_weight_edge_df = pd.DataFrame(bind_weight_src_dest)
# print('\n--------BIND-STREAM GENE EDGES WEIGHT--------')
# print(gene_bind_weight_edge_df)
# gene_bind_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv', index = False, header = True)
# # DOING AGGREGATION ON SAME SRC/DEST
# print('\n--------AGGBIND-STREAM GENE EDGES WEIGHT--------')
# gene_aggbind_weight_edge_df = gene_bind_weight_edge_df.groupby(['src', 'src_name', 'dest', 'dest_name']).agg({'weight': 'sum'}).reset_index()
# print(gene_aggbind_weight_edge_df)
# gene_aggbind_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_aggbind_weight_edge.csv', index = False, header = True)


# # CONVERT ADJACENT MATRICES WITH GENE OUT/IN DEGREE
# gene_list = []
# gene_name_list = []
# gene_outdeg_list = []
# gene_indeg_list = []
# gene_degree_list = []
# for idx in range(num_gene):
#     gene_list.append(idx + 1)
#     gene_name_list.append(cellline_gene_num_dict[idx + 1])
#     # ########ONLY USE FIRST CONV WEIGHT########
#     gene_outdeg = np.sum(first_conv_weight_adj[idx, :])
#     gene_outdeg_list.append(gene_outdeg)
#     # ########ONLY USE FIRST CONV WEIGHT########
#     gene_indeg = np.sum(first_conv_weight_adj[:, idx])
#     gene_indeg_list.append(gene_indeg)
#     gene_degree = gene_outdeg + gene_indeg
#     gene_degree_list.append(gene_degree)
# weight_gene_degree = {'gene_idx': gene_list, 'gene_name': gene_name_list, 'out_degree': gene_outdeg_list,\
#                 'in_degree': gene_indeg_list, 'degree': gene_degree_list}
# gene_weight_degree_df = pd.DataFrame(weight_gene_degree)
# print(gene_weight_degree_df)
# gene_weight_degree_df.to_csv('.' + dir_opt + '/analyse_data/gene_weight_degree.csv', index = False, header = True)