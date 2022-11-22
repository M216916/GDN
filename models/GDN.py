import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
import pandas as pd
import os
import pytorch_lightning as pl

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class Net(pl.LightningModule):

    def __init__(self, input_size=56, hidden_size=10, output_size=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, config={}):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()                                 # cpu
        edge_index = edge_index_sets[0]                       # tensor[[ 1, 2, 3,..., 23, 24, 25], [0, 0, 0,..., 26, 26, 26]] : (2,702)
        embed_dim = dim                                       # 64
        self.embedding = nn.Embedding(node_num, embed_dim)    # node_num : 27 ／ embed_dim : 64
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        
        self.gnn_layers = nn.ModuleList([GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)])
        self.node_embedding = None
        self.topk = topk                                      # 20
        self.learned_graph = None

#        self.neural_network = Net(pl.LightningModule)

        self.cache_edge_index_sets = [None] * edge_set_num    # [None]
        self.cache_embed_index = None                         # None
        self.dp = nn.Dropout(0.2)
        self.init_params()
        self.config = config

    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]                     # 702
            cache_edge_index = self.cache_edge_index_sets[i]   # tensor[[1, 2, 3, ..., 860, 861, 862], [0, 0, 0, ..., 863, 863, 863]]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]   # tensor[[1, 2, 3, ..., 860, 861, 862], [0, 0, 0, ..., 863, 863, 863]]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))    # torch.Size[ 27, 64]
            weights_arr = all_embeddings.detach().clone()                         # torch.Size[ 27, 64]
            all_embeddings = all_embeddings.repeat(batch_num, 1)                  # torch.Size[864, 64] (repeat*32 のとき)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)                      # torch.Size[864, 64]
        x = x.view(batch_num, node_num, -1)                 # torch.Size[32, 27, 64]

        indexes = torch.arange(0,node_num).to(device)       # tensor[ 0,  1,  2,  3, ... , 24, 25, 26] : torch.Size[27] →embedding→ [27, 64]
        
        out = torch.mul(x, self.embedding(indexes))         # torch.Size[32, 27, 64]     # out[i,j,k] = x[i,j,k] * emb[j,k]
        out = out.permute(0,2,1)                            # torch.Size[32, 64, 27]     # 要素入れ替え
        out = F.relu(self.bn_outlayer_in(out))              # torch.Size[32, 64, 27]     # Batch Normalization
        out = out.permute(0,2,1)                            # torch.Size[32, 27, 64]     # 要素入れ替え
        out = self.dp(out)                                  # torch.Size[32, 27, 64]     # drop out でニューロンを調整(過学習抑制)

        dataset = self.config['comment']

        x_non = pd.read_csv(f'./data/{dataset}/x_non.csv')
        x_non = torch.tensor(x_non.values, dtype=torch.int64)
        x_non = torch.t(x_non[:,1:]).float()
        x_non_list = []
        for i in range(out.shape[0]):
            x_non_list.append(x_non)
        x_non = torch.stack(x_non_list)

        out = torch.cat([out, x_non], dim=2)
        out = out.view(out.shape[0]*out.shape[1], out.shape[2])
        
        net = Net()
        out = net.forward(out)               # NN の出力 [320, 3]

        return out