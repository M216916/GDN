# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config        # batch:32, epoch:3, slide_win:5, dim:64, slide_stride:1, comment:msl, seed:5
                                                # out_layer_num:1, out_layer_inter_dim:128, decay: 0.0, val_ratio:0.2, topk:5
        self.env_config = env_config            # save_path:msl, dataset:msl, report:best, device:cpu, load_model_path: ''
        self.datestr = None

        dataset = self.env_config['dataset']                                              # msl
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)     # train : (1565,27)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)       # test  : (2049,28) columnに'attack'がある
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)                                            # M-6, M-1, M-2, S-2 … : len 27
        fc_struc = get_fc_graph_struc(dataset)                                            # M-6:[M-1, M-2, S-2 …], M-1:[M-6, M-2, S-2 …],  … : len 27

        set_device(env_config['device'])
        self.device = get_device()                                                        # cpu

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)                       # (2,702) : len 2   torch.int64に変換

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)                   # (28, 1565)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())  # (28, 2049)


        cfg = {
            'slide_win': train_config['slide_win'],                                           # slide_win    : 5
            'slide_stride': train_config['slide_stride'],                                     # slide_stride : 1
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)    
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        
             #【train】                                          #【test】
             #    [0][0]     [0][1]    [0][2]    [0][3]          #    [0][0]     [0][1]    [0][2]    [0][3]
             #    (27,5)      (27)       ()      (2,702)         #    (27,5)      (27)       ()      (2,702)
             #       …         …         …         …             #       …         …         …         …
             #       …         …         …         …             #       …         …         …         …
             # [1559][0]  [1559][1]  [1559][2]  [1559][3]        # [2043][0]  [2043][1]  [2043][2]  [2043][3]
             #    (27,5)      (27)       ()      (2,702)         #    (27,5)      (27)       ()      (2,702)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)

                                        # import pprint
                                        # pprint.pprint(vars(self.test_dataloader))
        
                                        # 【self.train_dataloader】【self.val_dataloader】【self.test_dataloader】
                                        # _DataLoader__initialized             : True
                                        # _DataLoader__multiprocessing_context : None
                                        # _IterableDataset_len_called          : None
                                        # _dataset_kind                        : 0
                                        # batch_sampler                        : <torch.utils.data.sampler.BatchSampler object at ~>
                                        # batch_size                           : 32
                                        # collate_fn                           : <function default_collate at ~>
                                        # dataset                              : <torch.utils.data.dataset.Subset object at ~>
                                        # drop_last                            : False
                                        # num_workers                          : 0
                                        # pin_memory                           : False
                                        # sampler                              : <torch.utils.data.sampler.RandomSampler object at ~>
                                        # timeout                              : 0
                                        # worker_init_fn                       : None

        
        

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)                               # edge_index_sets[0][0]:702 ／ [0][1]:702

        self.model = GDN(edge_index_sets, len(feature_map),                 # len(feature_map) = 27
                dim=train_config['dim'],                                    # 64
                input_dim=train_config['slide_win'],                        # 5
                out_layer_num=train_config['out_layer_num'],                # 1
                out_layer_inter_dim=train_config['out_layer_inter_dim'],    # 128
                topk=train_config['topk']                                   # 5
            ).to(self.device)                                               # cpu



    def run(self):

        if len(self.env_config['load_model_path']) > 0:                     # × (len(self.env_config['load_model_path'] = 0)
            model_save_path = self.env_config['load_model_path']            # ×

        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader
    

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])                                                       # 27
        np_test_result = np.array(test_result)                                                     # (3, 2044, 27)
        np_val_result = np.array(val_result)                                                       # (3,  312, 27)

        test_labels = np_test_result[2, :, 0].tolist()                                             # len : 2044  0/1 のみで構成
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)                  # test_scores   : (27, 2044)
                                                                                                   # normal_scores : (27,  312)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)               # len : 5
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)  # len : 5


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





