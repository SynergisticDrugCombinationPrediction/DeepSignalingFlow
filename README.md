# DeepSignalingFlow
Complex signaling pathways/networks are believed to be responsible for drug resistance in cancer therapy. And there are 2 challenges always being an open problem for effective drug combination prediction. First, the complex and biological meaningful gene regulatory relationships were not incorporated, and thus were not interpretable. Second, no model was designed to investigate complex mechanism of synergy (MoS). In this study, we proposed a novel computational model, DeepSignalingFlow, to investigate these 2 challenges. Specifically, a graph convolutional network (GCN) was built on a core cancer signaling network that consists of 1634 genes, with gene expression data, from 46 core cancer signaling pathways. Then the up-stream signaling-flow (from up-stream signaling to drug targets), and the down-stream signaling-flow (from drug targets to down-stream signaling) were mimicked by the trainable weights of network edges.  

<!-- For more details of DeepSignalingFlow, see our [bioRxiv paper]() -->


## 1. Data Preprocess
This study intergrates following datasets
* NCI ALMANAC drug combination screening dataset 
* Gene expression data of NCI-60 Cancer Cell Lines
* KEGG signaling pathways and cellular process
* Drug-Target interactions from DrugBank database

Finally, those datasets files will be parsed into numpy files to train our DeepSignalingFlow model.

```
python3 parse_file.py
```

## 2. Train Graph Bidirectional Convolutional Network (GBCN)
GBCN add weight on bidirectional adjacent matrices and predict synergistic drug scores thorugh decoder.
```
python3 tmain_bigraphsage_decoder.py
```
Besides, there are some other models to run

```
# GCN
python3 tmain_gcn.py

# GCN + decoder
python3 tmain_gcn_decoder.py

# GraphSAGE + decoder
python3 tmain_graphsage_decoder.py

# GAT(Graph Attention Network) + decoder
tmain_gat_decoder.py

# GraphSAGE + LSTM + decoder
tmain_graphsage_lstm_decoder.py
```