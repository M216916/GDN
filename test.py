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
    for x, y, labels, edge_index, x_non, true in dataloader:
        x, y, labels, edge_index, x_non, true = [item.to(device).float() for item in [x, y, labels, edge_index, x_non, true]] 
                        
        with torch.no_grad():
            out = model(x, edge_index, x_non)
            out = out.float().to(device)

            true = true.to(torch.int64)
            true = true.view(-1)  

            CE_loss = CE_loss_func(out, true)
        
        test_loss_list.append(CE_loss.item())
        acu_loss += CE_loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss