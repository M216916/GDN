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


def CE_loss_func(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)

def test(model, dataloader, config):

    device = get_device()

    test_loss_list = []
    now = time.time()

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]] 
                        
        with torch.no_grad():
            out = model(x, edge_index)
            out = out.float().to(device)

            dataset = config['comment']
            t = pd.read_csv(f'./data/{dataset}/true.csv')
            t = torch.tensor(t.values, dtype=torch.int64).squeeze()
            true = torch.cat([t,t], 0)
            for i in range(int(out.shape[0]/len(t))-2):
                true = torch.cat([true,t], 0)  

            CE_loss = CE_loss_func(out, true)
        
        test_loss_list.append(CE_loss.item())
        acu_loss += CE_loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss