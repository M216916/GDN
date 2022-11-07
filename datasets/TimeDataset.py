import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        
        self.raw_data = raw_data             # train : 28 list × 1565 ／ test : 28 list × 2049
        self.config = config                 # { slide_win : 5, slide_stride : 1}
        self.edge_index = edge_index         # tensor[[ 1,  2,  3,  ..., 23, 24, 25], [ 0,  0,  0,  ..., 26, 26, 26]] ... (2, 702)
        self.mode = mode                     # train ／ test
        x_data = raw_data[:-1]               # train : 27 list × 1565 ／ test : 27 list × 2049
        labels = raw_data[-1]                # train : 1565 (0 or 1)  ／ test : 2049  (0.0 or 1.0)
        data = x_data
        data = torch.tensor(data).double()      # torch.Size[27, 1565]   ／ torch.Size[27, 2049]     # to tensor
        labels = torch.tensor(labels).double()  # torch.Size[ 1, 1565]   ／ torch.Size[ 1, 2049]

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win, slide_stride = [self.config[k] for k    # slide_win : 5 ／ slide_stride : 1
            in ['slide_win', 'slide_stride']]
        is_train = self.mode == 'train'                    # is_train       : True ／ False
        node_num, total_time_len = data.shape              # node_num       :   27 ／   27      # total_time_len : 1565 ／ 2049
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
                                                           # rang           : range(5, 1565) ／ range(5, 2049)
        
        for i in rang:                                     # i   : 5, 6, ... , 1564  ／ 5, 6, ... , 2048
            ft = data[:, i-slide_win:i]                    # ft  : torch.Size[27, 5] ／ torch.Size[27, 5]
            tar = data[:, i]                               # tar : torch.Size[27]    ／ torch.Size[27]
            x_arr.append(ft)                               # len : 1560              ／ 2044
            y_arr.append(tar)                              # len : 1560              ／ 2044
            labels_arr.append(labels[i])                   # len : 1560              ／ 2044

        x = torch.stack(x_arr).contiguous()                # x[1559][26] : torch.Size[27, 5] ／ x[2043][26] : torch.Size[27, 5]
        y = torch.stack(y_arr).contiguous()                # y[1559][26] : torch.Size[27]    ／ y[2043][26] : torch.Size[27]
        labels = torch.Tensor(labels_arr).contiguous()     # labels[1559]: torch.Size[]      ／ labels[2043]: torch.Size[]
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()
        return feature, y, label, edge_index
