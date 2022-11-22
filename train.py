import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import pandas as pd



def CE_loss_func(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    sum_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    print(model.state_dict()['embedding.weight'])
    print(model.state_dict().keys())
    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        sum_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]               

            optimizer.zero_grad()
            out = model(x, edge_index)
            out = out.float().to(device)

            dataset = config['comment']
            t = pd.read_csv(f'./data/{dataset}/true.csv')
            t = torch.tensor(t.values, dtype=torch.int64).squeeze()
            true = torch.cat([t,t], 0)
            for i in range(int(out.shape[0]/len(t))-2):
                true = torch.cat([true,t], 0)        

            CE_loss = CE_loss_func(out, true)
            
            CE_loss.backward()
            optimizer.step()
            
            train_loss_list.append(CE_loss.item())              # loss値 を記録していく (39 * epoch数)
            sum_loss += CE_loss.item()                          # loss値 の1epochあたりの和 (→平均化する)
                
            i += 1

        print('epoch ({} / {}) (Loss:{:.8f})'.
            format(i_epoch, epoch, sum_loss/len(dataloader)), flush=True)

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss = test(model, val_dataloader, config)   # val_loss を出力

            if val_loss < min_loss:                              # val_loss の最小値が更新されなければ stop_improve_count +1 
                torch.save(model.state_dict(), save_path)        #  → early_stop_win に到達したら break
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:                                                    # ×
            if sum_loss < min_loss :                             # ×
                torch.save(model.state_dict(), save_path)        # ×
                min_loss = sum_loss                              # ×

    return train_loss_list
