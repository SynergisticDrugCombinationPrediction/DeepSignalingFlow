import os
import pdb
import torch
import argparse
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable

from load.read_graph import read_train_batch, read_test_batch
from load.graph_sampler import GraphLoader
from enc_dec.graphsage_lstm_decoder import GraphSAGELSTMDecoder

# PARSE ARGUMENTS FROM COMMAND LINE
def arg_parse():
    parser = argparse.ArgumentParser(description='GRAPHPOOL ARGUMENTS.')
    # ADD FOLLOWING ARGUMENTS
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--add-self', dest = 'adj_self',
                help = 'Graph convolution add nodes themselves.')
    parser.add_argument('--adj', dest = 'adj',
                help = 'Adjacent matrix is symmetry.')
    parser.add_argument('--model', dest = 'model',
                help = 'Model load.')
    parser.add_argument('--kfold', dest = 'kfold', type = float,
                help = 'Number of K-fold splits.')
    parser.add_argument('--lr', dest = 'lr', type = float,
                help = 'Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int,
                help = 'Batch size.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int,
                help = 'Number of epochs to train.')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int,
                help = 'Number of workers to load data.')
    parser.add_argument('--input-dim', dest = 'input_dim', type = int,
                help = 'Input feature dimension')
    parser.add_argument('--hidden-dim', dest = 'hidden_dim', type = int,
                help = 'Hidden dimension')
    parser.add_argument('--output-dim', dest = 'output_dim', type = int,
                help = 'Output dimension')
    parser.add_argument('--num-classes', dest = 'num_classes', type = int,
                help = 'Number of label classes')
    parser.add_argument('--num-gc-layers', dest = 'num_gc_layers', type = int,
                help = 'Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest = 'bn', action = 'store_const',
                const = False, default = True,
                help = 'Whether batch normalization is used')
    parser.add_argument('--dropout', dest = 'dropout', type = float,
                help = 'Dropout rate.')

    # SET DEFAULT INPUT ARGUMENT
    parser.set_defaults(cuda = '0',
                        add_self = '0', # 'add'
                        adj = '0', # 'sym'
                        model = '0', # 'load'
                        kfold_num = 5,
                        lr = 0.01,
                        clip= 2.0,
                        batch_size = 16,
                        num_epochs = 100,
                        num_workers = 2,
                        input_dim = 3,
                        hidden_dim = 3,
                        output_dim = 3,
                        decoder_dim = 10,
                        num_classes = 1,
                        num_gc_layer = 3,
                        dropout = 0.0)
    return parser.parse_args()

def learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3):
    epoch_iteration = int(dl_input_num / args.batch_size)
    l1 = (args.lr - 0.008) / (e1 * epoch_iteration)
    l2 = (0.008 - 0.006) / (e2 * epoch_iteration)
    l3 = (0.006 - 0.005) / (e3 * epoch_iteration)
    l4 = 0.005
    if iteration_num <= (e1 *epoch_iteration):
        learning_rate = args.lr - iteration_num * l1
    elif iteration_num <= (e1 + e2) * epoch_iteration:
        learning_rate = 0.008 - (iteration_num - e1 * epoch_iteration) * l2
    elif iteration_num <= (e1 + e2 + e3) * epoch_iteration:
        learning_rate = 0.006 - (iteration_num - (e1 + e2) * epoch_iteration) * l3
    else:
        learning_rate = l4
    print('-------LEARNING RATE ' + str(learning_rate) + '-------' )
    return learning_rate

def build_graphsage_lstm_model(args):
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    embedding_dim = args.output_dim
    decoder_dim = args.decoder_dim
    num_layer = args.num_gc_layer
    dropout = args.dropout
    graph_attribute_list, graph_adj_list, graph_inv_deg_list, graph_label_list = read_train_batch(0, 1, args)
    dataset_loader, node_num, feature_dim = GraphLoader.load_graph(graph_attribute_list,
                graph_adj_list, graph_inv_deg_list, graph_label_list, prog_args)
    model = GraphSAGELSTMDecoder(args.add_self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_layer, dropout)
    return model

def train_graphsage_lstm_model(dataset, model, args, learning_rate):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    for batch_idx, data in enumerate(dataset):
        optimizer.zero_grad()
        adj = Variable(data['adj'].float(), requires_grad = False)
        inv_deg = Variable(data['inv_deg'].float(), requires_grad = False)
        x = Variable(data['feature'].float(), requires_grad = False)
        label = Variable(data['label'].float())
        # THIS WILL USE METHOD [def forward()] TO MAKE PREDICTION
        ypred = model(x, adj, inv_deg)
        loss = model.loss(ypred, label)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        # pdb.set_trace()
    return model, loss, ypred

def train_graphsage_lstm(args):
    # BUILD [GraphSAGE, DECODER] MODEL
    model = build_graphsage_lstm_model(args)
    if args.model == 'load':
        model.load_state_dict(torch.load('./datainfo2/result/epoch_val_70/best_val_model-2-49.pth'))
    # TRAIN MODEL ON TRAINING DATASET
    dir_opt = '/datainfo2'
    form_data_path = '.' + dir_opt + '/form_data'
    xTr = np.load(form_data_path + '/xTr.npy')
    yTr = np.load(form_data_path + '/yTr.npy')
    dl_input_num = xTr.shape[0]
    epoch_num = args.num_epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    # RECORD EPOCH LOSS AND PEARSON CORRELATION
    iteration_num = 0
    min_train_loss = 100
    min_train_id = 0
    e1 = 20
    e2 = 20
    e3 = 10
    epoch_loss_list = []
    epoch_pearson_list = []
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    folder_name = 'epoch_' + str(epoch_num)
    path = '.' + dir_opt + '/result/%s' % (folder_name)
    unit = 1
    while os.path.exists('.' + dir_opt + '/result') == False:
        os.mkdir('.' + dir_opt + '/result')
    while os.path.exists(path):
        path = '.' + dir_opt + '/result/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)
    for i in range(1, epoch_num + 1):
        print('------------------------EPOCH: ' + str(i) + ' ------------------------')
        model.train()
        epoch_ypred = np.zeros((1, 1))
        upper_index = 0
        batch_loss_list = []
        count = 0
        last_weight = 0
        for index in range(0, dl_input_num, batch_size):
            if (index + batch_size) < dl_input_num:
                upper_index = index + batch_size
            else:
                upper_index = dl_input_num
            graph_attribute_list, graph_adj_list, graph_inv_deg_list, graph_label_list = read_train_batch(index, upper_index, args)
            dataset_loader, node_num, feature_dim = GraphLoader.load_graph(graph_attribute_list,
                        graph_adj_list, graph_inv_deg_list, graph_label_list, prog_args)
            # ACTIVATE LEARNING RATE SCHEDULE
            iteration_num += 1
            learning_rate = learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3)
            # learning_rate = 0.005
            model, batch_loss, batch_ypred = train_graphsage_lstm_model(dataset_loader, model, args, learning_rate)
            batch_loss = batch_loss.item()
            print('BATCH LOSS: ', batch_loss)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('EPOCH ' + str(i) + ' MSE LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        print(epoch_ypred)
        # PRESERVE PEARSON CORR FOR EVERY EPOCH
        score_lists = list(yTr)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'Score': score_list, 'Pred Score': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # pdb.set_trace()
        epoch_pearson = tmp_training_input_df.corr(method = 'pearson')
        epoch_pearson_list.append(epoch_pearson['Pred Score'][0])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index = False, header = True)
        print('EPOCH ' + str(i) + ' PEARSON CORRELATION: ', epoch_pearson)
        # SAVE BEST TRAINING AND VALIDATION MODEL
        if epoch_loss < min_train_loss:
            min_train_loss = epoch_loss
            min_train_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pth')
            torch.save(model.state_dict(), path + '/best_train_model.pth')
        print('\n-------------BEST MODEL ID:' + str(min_train_id) + '-------------')
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



def test_graphsage_lstm_model(dataset, model, args):
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad = False)
        inv_deg = Variable(data['inv_deg'].float(), requires_grad = False)
        x = Variable(data['feature'].float(), requires_grad = False)
        label = Variable(data['label'].float())
        # THIS WILL USE METHOD [def forward()] TO MAKE PREDICTION
        ypred = model(x, adj, inv_deg)
        loss = model.loss(ypred, label)
        # pdb.set_trace()
    return model, loss, ypred


def test_graphsage_lstm(args):
    # BUILD [GraphSAGE, DECODER] MODEL
    model = build_graphsage_model(args)
    model.load_state_dict(torch.load('./datainfo2/result/epoch_100/best_train_model.pth'))
    # TEST MODEL ON TRAINING DATASET
    dir_opt = '/datainfo2'
    form_data_path = '.' + dir_opt + '/form_data'
    xTe = np.load(form_data_path + '/xTe.npy')
    yTe = np.load(form_data_path + '/yTe.npy')
    dl_input_num = xTe.shape[0]
    batch_size = args.batch_size
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    path = '.' + dir_opt + '/result/epoch_100'
    # RUN TEST MODEL
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, batch_size):
        if (index + batch_size) < dl_input_num:
            upper_index = index + batch_size
        else:
            upper_index = dl_input_num
        graph_attribute_list, graph_adj_list, graph_inv_deg_list, graph_label_list = read_test_batch(index, upper_index, args)
        dataset_loader, node_num, feature_dim = GraphLoader.load_graph(graph_attribute_list,
                    graph_adj_list, graph_inv_deg_list, graph_label_list, prog_args)
        # ACTIVATE LEARNING RATE SCHEDULE
        model, batch_loss, batch_ypred = test_graphsage_lstm_model(dataset_loader, model, args)
        batch_loss = batch_loss.item()
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # PRESERVE PREDICTION OF BATCH TRAINING DATA
        batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('MSE LOSS: ', test_loss)
    # PRESERVE PEARSON CORR FOR EVERY EPOCH
    all_ypred = np.delete(all_ypred, 0, axis = 0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'Score': score_list, 'Pred Score': all_ypred_list}
    tmp_test_input_df = pd.DataFrame(test_dict)
    test_pearson = tmp_test_input_df.corr(method = 'pearson')
    tmp_test_input_df.to_csv(path + '/TestPred.txt', index = False, header = True)
    print('PEARSON CORRELATION: ', test_pearson)
    

if __name__ == "__main__":
    # PARSE ARGUMENT FROM TERMINAL OR DEFAULT PARAMETERS
    prog_args = arg_parse()
    # CHECK CUDA GPU DEVICES ON MACHINE
    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
    train_graphsage_lstm(prog_args)
    # test_graphsage_lstm(prog_args)
