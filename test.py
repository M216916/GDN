import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *



def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)      # val : 10 (312 ÷ batch_size 32) ／ test : 64 (2043 ÷ batch_size 32)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
                                                                        # x          : torch.Size[32, 27, 5]
                                                                        # y          : torch.Size[32, 27]
                                                                        # labels     : torch.Size[32]
                                                                        # edge_index : torch.Size[32, 2, 702]        
        with torch.no_grad():
            out_1, predicted = model(x, edge_index)
            predicted = predicted.float().to(device)
                                                                        # predicted  : torch.Size[32, 27]
            loss = loss_func(predicted, y)
            
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])  # torch.Size[32, 27]
                                                                        # [32] →unsqueeze(1)→ [32, 1] →repeat(1,27)→ [32, 27]

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)   # torch.Size[32, 27] → [64, 27] → ... → [312, 27]
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)                 # torch.Size[32, 27] → [64, 27] → ... → [312, 27]
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)            # torch.Size[32, 27] → [64, 27] → ... → [312, 27]
                
                
        
        test_loss_list.append(loss.item())                   # len ... val : 10 ／ test : 64
        acu_loss += loss.item()                              # loss の和
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)       # loss の平均 (val での loss ／test での loss を出力)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




