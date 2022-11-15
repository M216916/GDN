import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer

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

import json
import random

import warnings
warnings.filterwarnings('ignore')

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {'slide_win': train_config['slide_win'], 
               'slide_stride': train_config['slide_stride'],}

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)    
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk'] ).to(self.device) 


    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']

        else:
            model_save_path = self.get_save_path()[0]                       # ./pretrained/msl/best_09|21-17:31:58.pt

            self.train_log = train(self.model, model_save_path, 
                config = train_config,                                      # batch:32, epoch:3, slide_win:5, dim:64, slide_stride:1, comment:msl, ...
                train_dataloader=self.train_dataloader,                     # torch.utils.data.dataloader.DataLoader object
                val_dataloader=self.val_dataloader,                         # torch.utils.data.dataloader.DataLoader object
                feature_map=self.feature_map,                               # M-6, M-1, M-2, S-2 … : len 27
                test_dataloader=self.test_dataloader,                       # torch.utils.data.dataloader.DataLoader object
                test_dataset=self.test_dataset,                             # datasets.TimeDataset.TimeDataset object
                train_dataset=self.train_dataset,                           # datasets.TimeDataset.TimeDataset object
                dataset_name=self.env_config['dataset'] )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result, conv_list = test(best_model, self.test_dataloader)
        _, self.val_result, __ = test(best_model, self.val_dataloader)

        dataset = self.env_config['dataset']
        dim = self.train_config['dim']
        t = pd.read_csv(f'./data/{dataset}/true.csv')
        t = t.transpose().rename(columns={0: 'target'})

        x_2 = pd.read_csv(f'./data/{dataset}/x_non.csv')
        x_2 = x_2.transpose()

        list_embed = list(range(x_2.shape[1], x_2.shape[1]+dim, 1))
        x_1 = conv_list[-1][-1]
        x_1 = x_1.to('cpu').detach().numpy().copy()
        x_1 = pd.DataFrame(x_1, columns = list_embed)
        x_1.index = t.index     

        data = pd.concat([x_2, x_1, t], axis=1)

        self.LightGBM(data)

        self.get_score(self.test_result, self.val_result)


    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))                               # 1560
        train_use_len = int(dataset_len * (1 - val_ratio))                  # 1248
        val_use_len = int(dataset_len * val_ratio)                          # 312
        val_start_index = random.randrange(train_use_len)                   # 523
        indices = torch.arange(dataset_len)                                 

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)
        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader
    

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)
 
        GDN_num = os.getcwd().replace('/home/inaba/', '')
        dataset = self.env_config['dataset']

        folder_path = f'/home/inaba/GDN_img/{GDN_num}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        folder_path = f'/home/inaba/GDN_img/{GDN_num}/{dataset}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        

        for i in range(np_test_result.shape[2]):
            fig = plt.figure()
            plt.plot(np_test_result[0,:,i], label='Prediction')
            plt.plot(np_test_result[1,:,i], label='GroundTruth')
            plt.legend()
            plt.show()
            fig.savefig(folder_path + "img_" + str(i) + ".png")


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


    def LightGBM(self, data):
        from sklearn.model_selection import train_test_split
        import lightgbm as lgb

        RANDOM_STATE = 10        # ランダムシード値（擬似乱数）
        TEST_SIZE = 0.2          # 学習データと評価データの割合

        # 学習データと評価データを作成
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:data.shape[1]-1],
                                                            data.iloc[:, data.shape[1]-1],
                                                            test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE)

        # trainのデータセットの2割をモデル学習時のバリデーションデータとして利用する
        x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                            y_train,
                                                            test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE)

        # LightGBMを利用するのに必要なフォーマットに変換
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)        

        params = {'objective' : 'multiclass',
                  'metric' : 'multi_logloss',
                  'num_class' : 3,
                  'depth':1,
                  'learning_rate': 0.3,}

        # LightGBM学習
        evaluation_results = {}
        evals = [(lgb_train, 'train'), (lgb_eval, 'eval')]

        model = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_names = evals,
                        valid_sets = [lgb_train, lgb_eval],
                        early_stopping_rounds=20)

        pred = model.predict(x_test, num_iteration = model.best_iteration)
        label = pred.argmax(axis = 1)
        print("=" * 100)
        print('▼pred\n', label)
        print("=" * 100)
        print('▼true\n',y_test.values)

        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(y_test, label,labels = [0,1,2])
        print(matrix)

        accuracy = np.trace(matrix)/np.sum(matrix)
        print('accuracy      :{0:4f}'.format(accuracy))

        for i in range(3):
            precision = matrix[i,i]/np.sum(matrix[:,i])
            recall = matrix[i,i]/np.sum(matrix[i,])
            F1 = (2*precision * recall)/(precision + recall)
            print('【{0}】precision:{1:4f}  recall:{2:4f}  F1:{3:4f}'.format(i,precision,recall,F1))



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