# Graph-Conv
Basic Graph Convolution

## 1 Prepare Data
### 1.1 Pre-Manual to Form Drug Map
#### input_condense()
#### drug_map()
* Average multiple dose for score in ['init_data/DeepLearningInput.csv'] => 'mid_data/DeepLearningInput.txt'
* Find common drug names in input file ['mid_data/DeepLearningInput.txt'] and drug_tar file ['init_data/drug_tar_drugBank_all.txt']
* Auto save common drug name map file => 'init_data/drug_map.csv'
* Manually match common drug name map file => *['mid_data/drug_map.csv']*

### 1.2.1 Pre-Parse [Drug-Target] Matrix for [ALL 5435 DRUGS <-> ALL 2775 GENES]
#### drug_map_dict()
* Auto save manual drug name map file ['mid_data/drug_map.csv'] => *'/filtered_data/drug_map_dict.npy'*
#### drug_target() 
* From ['/init_data/drug_tar_drugBank_all.txt'] form unique drug name-num dict => *'/filtered_data/drug_dict.npy'*
* From ['/init_data/drug_tar_drugBank_all.txt'] form unique gene name-num dict => *'/filtered_data/target_dict.npy'*
* Form [Drug-Target] Matrix *'/filtered_data/drug_target_matrix.npy'*

### 1.2.2 Pre-Parse to Form Cellname Map
#### cellline_map_dict()
* From ['/init_data/nci60-ccle_cell_name_map1.txt'] => *'/filtered_data/cellline_map_dict.npy'*

### 1.2.3 Pre-Parse to Filter Unqualified Genes RNA in Cellline and Genes RNA not Exist in Edges File
#### filter_cellline_gene(RNA_seq_filename)
* From ['init_data/nci60-ccle_RNAseq_tpm2.txt'] => 'mid_data/nci60-ccle_RNAseq_tpm2.csv'
#### filter_form_edge_cellline_gene()
* From 'mid_data/nci60-ccle_RNAseq_tpm2.csv' to filter genes not in ['init_data/Selected_Kegg_Pathways_edges_1.txt']
* Save filtered genes RNA_seq to *'filtered_data/nci60-ccle_RNAseq_tpm2.csv'*

### 1.2.4 Pre-Parse to Build Num Map Between [Cellline RNA 1634 Genes] and [Drug-Target Genes]
#### gene_target_num_dict(RNA_seq_filename)
* From 'filtered_data/nci60-ccle_RNAseq_tpm2.csv' and '/filtered_data/target_dict.npy' 
* Save num map for Gene-Target number map => '/filtered_data/gene_target_num_dict.npy'

### 1.2.5 Pre-Parse [Cellline RNA Genes - Pathway] Matrix for [2142 Genes <-> 46 Pathways]
#### gene_pathway()
* Pathway file also has unique genes, should be consistent with Cellline RNA Genes
* From ['init_data/Selected_Kegg_Pathways2.txt']
* => *'./datainfo2/filtered_data/gene_pathway_matrix.npy'*
* => *'./datainfo2/filtered_data/gene_dict.npy'*
* => *'./datainfo2/filtered_data/pathway_dict.npy'*


### 1.3 Pre-Input to Condense
* input_condense(): Average Dose
* input_drug_condense(): Remove No Map Drug Name Points
* input_cellline_condense(): Remove No Cellline Name Points
* input_drug_gene_condense(): Remove Gene which is not Target of Drug_A or Drug_B

### 1.4 Pre-Form
* LoadData.load_all()


## 2 Run Different Model
```
    /encoder_test
    # datainfo2 - Graph-Conv/t-gcn (torch.max() pooling and concat)
    python3 tmain_gcn.py

    /decoder_test
    # datainfo2 - Graph-Conv/t-gcndecoder (no pooling and no graphsage with drug decoder)
    python3 tmain_gcn_decoder.py

    # datainfo2 - Graph-Conv/t-graphsage (graphsage with adj using weight_matrix and drug decoder)
    python3 ttmain_graphsage_decoder.py

    # datainfo2 - Graph-Conv/t-gat (graphsage with adj using gat attention way and drug decoder)
    python3 tmain_gat_decoder.py
```

## 3 Result Summary About Different Models
### 3.1 Toy Models
* Gcn + max-pooling: 
    * Training Dataset
        * MSE Loss: 75
        * Pearson Correlation: 0.08

* Gcn + decoder (no pooling): 
    * Training Dataset
        * MSE Loss: 70
        * Pearson Correlation: 0.30

* GraphSAGE + decoder (no pooling): 
    * Training Dataset
        * MSE Loss: 60
        * Pearson Correlation: 0.45

### 3.2 Combined Toy Models
* GraphSAGE + Weight_adj(Includes Drugs) + decoder:
    * Training Dataset:
        * MSE Loss: 40
        * Pearson Correlation: 0.69
    * Test Dataset:
        * MSE Loss: 
        * Pearson Correlation: 0.626

* GraphSAGE + Weight_adj(No Drugs) + decoder:
    * Training Dataset:
        * MSE Loss: 49
        * Pearson Correlation: 0.62
    * Test Dataset:
        * MSE Loss: 
        * Pearson Correlation: 0.55

* GraphSAGE + GAT(Include Drugs) + decoder:
    * Training Dataset:
        * MSE Loss:
        * Pearson Correlation:
    * Test Dataset:
        * MSE Loss:
        * Pearson Correlation:

* GraphSAGE + GAT(No Drugs) + decoder:
    * Training Dataset:
        * MSE Loss:
        * Pearson Correlation:
    * Test Dataset:
        * MSE Loss:
        * Pearson Correlation: