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




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def CE_loss_func(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
                                                             # x          : torch.Size[32, 27, 5]
                                                             # labels     : torch.Size[32, 27]
                                                             # edge_index : torch.Size[32, 2, 702]
                        
#            x_ave = torch.mean(input=x, dim=2)
#            for i in range(x.shape[2]):
#                x[:,:,i] = x[:,:,i] / x_ave                        

            optimizer.zero_grad()
            out_1, out_2 = model(x, edge_index)
            out_2= out_2.float().to(device)

#            print('▼x', x.shape)
#            print('▼edge_index', edge_index.shape)
#            print('▼out_2', out_2.shape)
#            print('▼labels', labels.shape)

#            out_2 = out_2 * x_ave


            t = pd.read_csv('./data/yfinance_8/true.csv')
            t = torch.tensor(t.values, dtype=torch.int64).squeeze()
            true = torch.cat([t,t], 0)
            for i in range(int(out_1.shape[0]/len(t))-2):
                true = torch.cat([true,t], 0)        


#            loss = loss_func(out_2, labels)                  # MSE loss
            CE_loss = CE_loss_func(out_1, true)
            
            CE_loss.backward()
            optimizer.step()

            
            train_loss_list.append(CE_loss.item())              # loss値 を記録していく (39 * epoch数)
            acu_loss += CE_loss.item()                          # loss値 の1epochあたりの和 (→平均化する)
                
            i += 1


        # each epoch                                         # epoch ごとにloss値の平均を出力
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result, _ = test(model, val_dataloader)   # val_loss を出力

            if val_loss < min_loss:                              # val_loss の最小値が更新されなければ stop_improve_count +1 
                torch.save(model.state_dict(), save_path)        #  → early_stop_win に到達したら break
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:                                                    # ×
            if acu_loss < min_loss :                             # ×
                torch.save(model.state_dict(), save_path)        # ×
                min_loss = acu_loss                              # ×



    return train_loss_list
