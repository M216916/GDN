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
from models.GDN import pre_GDN
from models.GDN import fin_GDN
from models.GDN import fin_GDN_onlytime
from models.GDN import fin_GDN_nontime
from train import pre_training
from train import fine_tuning
from test  import pre_test
from test  import fin_test
from train_boost import embedded_out
from train_boost import lgb_training
from train_boost import xgb_training
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
import sys
from datetime import datetime
import os
import argparse
from pathlib import Path
import json
import random
import math
import warnings
warnings.filterwarnings('ignore')


class Main():
    def __init__(self, train_config, env_config, debug=False, model_flag=''):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.model_flag = model_flag

        dataset = self.env_config['dataset']
        train = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)

###########################################################################
        ave_span = 5
        raw_num = len(train) // ave_span
        train_ = train.iloc[:raw_num, :]
        for i in range(train_.shape[0]):
            for j in range(train_.shape[1]):
                train_.iloc[i,j] = train.iloc[i*ave_span:(i+1)*ave_span, j].mean()
        train = train_
        print('***train', train.shape)
###########################################################################

        x_non = pd.read_csv(f'./data/{dataset}/x_non.csv', sep=',', index_col=0)
        x_non = x_non.apply(lambda x: (x-x.mean())/x.std(), axis=0)               #属性(columns)ごとに標準化
#        important = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 25, 29, 31, 34]
#        x_non = x_non.iloc[important,:]
        
        pre_train = train.iloc[    :2400 + train_config['slide_win'],:]
        fin_train = train.iloc[2000:2400 + train_config['slide_win'],:]
        fin_test  = train.iloc[                              2400-1:,:]

        pre_train = train.iloc[    : 450 + train_config['slide_win'],:]
        fin_train = train.iloc[ 300: 450 + train_config['slide_win'],:]
        fin_test  = train.iloc[                               450-1:,:]


##################################################################################################
        '''
        # fin_test の各クラスの数を数える
        classes = torch.zeros(fin_test.shape[0]-train_config['slide_win'], fin_test.shape[1])
        line = 0.009
        for i in range(classes.shape[0]):
            for j in range(classes.shape[1]):
                a = fin_test.iloc[i+train_config['slide_win']-1, j]
                b = fin_test.iloc[i+train_config['slide_win'],   j]
                rate = (b-a)/a
                if rate >= -line:
                    classes[i,j] = 1
                    if rate > line:
                        classes[i,j] = 2
        _, class_num = torch.unique(classes, return_counts=True)
        print(class_num[0])
        print(class_num[1])
        print(class_num[2])
        '''
##################################################################################################


        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.feature_map = feature_map

        cfg = {'slide_win': train_config['slide_win'],
               'slide_stride': train_config['slide_stride'],}


        # pre_training data

        pre_train_dataset_indata = construct_data(pre_train, feature_map, labels=0)

        pre_train_dataset = TimeDataset(pre_train_dataset_indata, fc_edge_index, mode='train', config=cfg, x_non=x_non, flag='pre')   
        pre_train_dataloader, pre_val_dataloader = self.get_loaders(
            pre_train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.pre_train_dataloader = pre_train_dataloader
        self.pre_val_dataloader = pre_val_dataloader        


        # fine_tunig data

        fin_train_dataset_indata = construct_data(fin_train, feature_map, labels=0)
        fin_test_dataset_indata  = construct_data(fin_test,  feature_map, labels=0)
        
        fin_train_dataset = TimeDataset(fin_train_dataset_indata, fc_edge_index, mode='train', config=cfg, x_non=x_non, flag='fin')
        fin_test_dataset  = TimeDataset(fin_test_dataset_indata, fc_edge_index, mode='train', config=cfg, x_non=x_non, flag='fin')

        fin_train_dataloader, fin_val_dataloader = self.get_loaders(
            fin_train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])
        fin_test_dataloader = DataLoader(fin_test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)

        self.fin_train_dataloader = fin_train_dataloader
        self.fin_val_dataloader = fin_val_dataloader
        self.fin_test_dataloader = fin_test_dataloader


        self.pre_model = pre_GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']).to(self.device)

        self.fin_model = fin_GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                dim_non=len(x_non),
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']).to(self.device)

        if self.model_flag=='onlytime':
            self.fin_model = fin_GDN_onlytime(edge_index_sets, len(feature_map),
                    dim=train_config['dim'],
                    dim_non=len(x_non),
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk']).to(self.device)

        if self.model_flag=='nontime':
            self.fin_model = fin_GDN_nontime(edge_index_sets, len(feature_map),
                    dim=train_config['dim'],
                    dim_non=len(x_non),
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk']).to(self.device)


    def run(self):

        print('■■■■■', self.model_flag, '■■■■■')
        print('▼▼▼ Pre-training : regression')

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']

        else:
            pre_model_save_path = self.pre_get_save_path()[0]   # モデル格納するpath生成

            self.train_log = pre_training(self.pre_model, pre_model_save_path, 
                config = train_config,
                train_dataloader=self.pre_train_dataloader,
                val_dataloader=self.pre_val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=None,
                test_dataset=None,
                train_dataset=None,
                dataset_name=self.env_config['dataset'])

        pre_load_weights = torch.load(pre_model_save_path)
        self.pre_model.load_state_dict(torch.load(pre_model_save_path))
        best_model = self.pre_model.to(self.device)

        _, self.val_result = pre_test(best_model, self.pre_val_dataloader)

        self.get_figure(self.val_result)


########################################################################################################
        if self.model_flag=='lgb':
            print('▼▼▼ Boosting : classifier')
            lgb_train, lgb_val, lgb_test = embedded_out(best_model, self.fin_train_dataloader, self.fin_val_dataloader, self.fin_test_dataloader)
            lgb_training(lgb_train, lgb_val, lgb_test)        


        elif self.model_flag=='xgb':
            print('▼▼▼ Boosting : classifier')
            xgb_train, xgb_val, xgb_test = embedded_out(best_model, self.fin_train_dataloader, self.fin_val_dataloader, self.fin_test_dataloader)
            xgb_training(xgb_train, xgb_val, xgb_test)


        else:
            print('▼▼▼ Fine-tuning : classifier')

            fin_model_save_path = self.fin_get_save_path()[0]
            fin_load_weights = self.fin_model.state_dict()
            for i in list(pre_load_weights.keys())[:17]:
                fin_load_weights[i] = pre_load_weights[i]
            self.fin_model.load_state_dict(fin_load_weights)

            # 凍結(freeze)の場合実行
            if self.model_flag=='freeze':
                for i in range(11):
                    list(self.fin_model.parameters())[i].requires_grad = False

            fine_tuning(self.fin_model, fin_model_save_path, 
                    config = train_config,
                    train_dataloader=self.fin_train_dataloader,
                    val_dataloader=self.fin_val_dataloader,
                    feature_map=self.feature_map,
                    test_dataloader=None,
                    test_dataset=None,
                    train_dataset=None,
                    dataset_name=self.env_config['dataset'])

            self.fin_model.load_state_dict(torch.load(fin_model_save_path))
            best_model = self.fin_model.to(self.device)

            Loss = fin_test(best_model, self.fin_test_dataloader, train_config, 'test')
            print('Loss:', Loss)




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

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader


    def get_figure(self, val_result):

        feature_num = len(val_result[0][0])
        np_val_result = np.array(val_result)
        
        GDN_num = os.getcwd().replace('/home/inaba/', '')
        dataset = self.env_config['dataset']
        folder_path = f'/home/inaba/GDN_img/{GDN_num}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path = f'/home/inaba/GDN_img/{GDN_num}/{dataset}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        

        for i in range(np_val_result.shape[2]):
            fig = plt.figure()
            plt.plot(np_val_result[0,:,i], label='Prediction')
            plt.plot(np_val_result[1,:,i], label='GroundTruth')
            plt.legend()
            plt.show()
            fig.savefig(folder_path + "img_" + str(i) + ".png")


    def pre_get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [f'./pretrained/{dir_path}/best_{datestr}.pt',
                 f'./results/{dir_path}/{datestr}.csv',]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


    def fin_get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [f'./finetuning/{dir_path}/best_{datestr}.pt',
                 f'./results/{dir_path}/{datestr}.csv',]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-pre_epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-fin_epoch', help='train epoch', type = int, default=100)
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
    parser.add_argument('-loss_function', help='loss_function', type = str, default='CE_loss')
    parser.add_argument('-Dice_gamma', help='Dice_gamma', type = float, default=0)
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    '''
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    '''

    train_config = {
        'batch': args.batch,
        'pre_epoch': args.pre_epoch,
        'fin_epoch': args.fin_epoch,
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
        'loss_function': args.loss_function,
        'Dice_gamma': args.Dice_gamma,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False, model_flag='full')
    main.run()

#    main = Main(train_config, env_config, debug=False, model_flag='freeze')
#    main.run()

#    main = Main(train_config, env_config, debug=False, model_flag='onlytime')
#    main.run()

#    main = Main(train_config, env_config, debug=False, model_flag='nontime')
#    main.run()

#    main = Main(train_config, env_config, debug=False, model_flag='xgb')
#    main.run()

#    main = Main(train_config, env_config, debug=False, model_flag='lgb')
#    main.run()


#     model_flag = ['full', 'freeze', 'onlytime', 'nontime', 'xgb', 'lgb']